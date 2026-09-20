"""Replace only MCTS leaf backup values using exactly replayed observations.

Privileged diagnostic, not a deployable controller. Production MCTS is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_failure_branches import (capture_root, load_model, make_env,
    make_search, monitor_at, source_data, summarize_rows)
from scripts.diagnose_stall_pulses import replay_to_branch
from scripts.diagnose_value_commitment import check_baseline, old_branch
from scripts.eval_controller_diagnostic import (atomic_json, atomic_npz, canonical_sha,
    file_sha, numpy_search_rng, observation_uint8, x_position)


def obs_sha(obs):
    return hashlib.sha256(observation_uint8(obs).tobytes()).hexdigest()


def leaf_path(node):
    moves = []
    while node.parent is not None:
        moves.append(int(node.move))
        node = node.parent
    return tuple(reversed(moves))


def leaf_search(search, obs, net, temperature, lookup, replace_values):
    """Hook only the scalar passed to Node.backup; retain original search code."""
    from src.muzero.node import Node
    original = Node.backup
    events = []

    def backup(node, value, config, minmax):
        if node.parent is None:
            return original(node, value, config, minmax)
        path = leaf_path(node)
        # Emulator observation queries must not consume search tie-break RNG.
        state = np.random.get_state()
        try:
            actual = lookup(path)
        finally:
            np.random.set_state(state)
        observed = 0. if actual['terminal'] else float(actual['value'])
        if not np.isfinite(observed):
            raise ValueError('Nonfinite observed leaf value')
        used = observed if replace_values else float(value)
        events.append({'path': list(path), 'depth': len(path), 'imagined_value': float(value),
                       'observed_value': observed, 'backed_up_value': used,
                       'predicted_reward': float(node.rwd), 'terminal': actual['terminal'],
                       'terminal_depth': actual['terminal_depth'],
                       'observation_sha256': actual['observation_sha256']})
        return original(node, used, config, minmax)

    with capture_root() as roots, patch.object(Node, 'backup', backup):
        result = search.run(obs, net, temperature=temperature, deterministic=False)
    if len(events) != search.num_simulations or len(roots) != 1:
        raise ValueError('Unexpected backup or root count')
    root = roots[0]
    return result, {'action': int(result[0]), 'visits': result[1].tolist(),
        'root_q': float(result[2]), 'root_prior': [float(c.prior) for c in root.children],
        'child_returns': [float(c.rwd+search.discount*c.Q) if c.N else None for c in root.children],
        'leaf_backups': events, 'max_depth': max(e['depth'] for e in events),
        'unique_paths': len({tuple(e['path']) for e in events})}


def same_search(first, second):
    if first[0] != second[0] or first[2] != second[2] or not np.array_equal(first[1], second[1]):
        raise ValueError('Observe-only instrumentation changed search')


def branch_result(seed, duration, decisions, outcome):
    if not isinstance(outcome['decisions'], int):
        raise ValueError('Rollout length must be an integer')
    return {**outcome, 'seed': seed, 'intervention_decisions': duration,
            'search_decisions': decisions}


def replay_original_root(env, record, trace, step):
    # The worker reuses one emulator; restore the random-start generator to the
    # state a newly constructed environment would have for this source episode.
    env.rng = np.random.default_rng(seed=record['env_seed'])
    return replay_to_branch(env, record, trace, step)


def observation_worker(connection, cfg, spec, record, trace):
    """Separate process permits two emulator instances without snapshot ambiguity."""
    env = None
    try:
        env = make_env(cfg, spec['case']['level'], record)
        while True:
            actions = connection.recv()
            if actions is None:
                break
            obs = replay_original_root(env, record, trace, spec['root']['step'])
            done, terminal_depth = False, None
            rewards = []
            for depth, action in enumerate(actions, 1):
                obs, reward, done, info = env.step(int(action))
                rewards.append(float(reward))
                if done:
                    terminal_depth = depth
                    break
            connection.send({'obs': obs, 'terminal': bool(done),
                             'terminal_depth': terminal_depth, 'rewards': rewards,
                             'observation_sha256': obs_sha(obs),
                             'x': x_position(env.last_info),
                             'player_state': int(env.last_info['player_state'])})
    except BaseException:
        import traceback
        connection.send({'error': traceback.format_exc()})
    finally:
        if env is not None:
            env.close()
        connection.close()


class ObservationValues:
    def __init__(self, cfg, spec, record, trace, net, saved=None):
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=observation_worker, args=(child,cfg,spec,record,trace))
        self.process.start()
        child.close()
        self.net = net
        self.cache = {} if saved is None else {tuple(row['actions']): row['result'] for row in saved}

    def get(self, actions):
        import torch
        key = tuple(int(a) for a in actions)
        if key not in self.cache:
            self.connection.send(key)
            if not self.connection.poll(300):
                raise TimeoutError('Emulator observation worker timed out')
            row = self.connection.recv()
            if 'error' in row:
                raise RuntimeError(row['error'])
            obs = row.pop('obs')
            if row['terminal']:
                row['value'] = 0.
            else:
                with torch.inference_mode():
                    _, _, value = self.net.initial_inference(torch.from_numpy(obs).to('cuda', dtype=torch.float32).unsqueeze(0))
                row['value'] = float(value.item())
            self.cache[key] = row
        return self.cache[key]

    def export(self):
        return [{'actions': list(actions), 'result': result} for actions,result in self.cache.items()]

    def close(self):
        if self.process.is_alive():
            try:
                self.connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
        self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
        self.connection.close()


def prepare(root, manifest):
    prior = Path(manifest['value_experiment'])
    for name, digest in manifest['value_file_sha256'].items():
        if file_sha(prior/name) != digest:
            raise ValueError(f'Value experiment changed: {name}')
    original = json.loads((prior/'manifest.json').read_text())
    plan = json.loads((prior/'plan.json').read_text())
    if not json.loads((prior/'summary.json').read_text())['complete']:
        raise ValueError('Source experiment incomplete')
    if plan['manifest_sha256'] != canonical_sha(original):
        raise ValueError('Source plan mismatch')
    for key in ('branch_experiment','branch_seeds','max_steps','num_simulations','leaf_batch','root_exploration_eps'):
        if manifest[key] != original[key]:
            raise ValueError(f'Paired setting changed: {key}')
    for level in manifest['levels']:
        if manifest['levels'][level]['checkpoint_sha256'] != original['levels'][level]['checkpoint_sha256']:
            raise ValueError('Paired checkpoint changed')
    current = Path(__file__).resolve().parents[1]
    for path in (prior/'source/src').rglob('*.py'):
        if file_sha(path) != file_sha(current/path.relative_to(prior/'source')):
            raise ValueError(f'Production source changed: {path}')
    for name in ('diagnose_failure_branches.py','diagnose_stall_pulses.py',
                 'eval_controller_diagnostic.py','stall_sampling_controller.py'):
        if file_sha(prior/'source/scripts'/name) != file_sha(current/'scripts'/name):
            raise ValueError(f'Paired helper changed: {name}')
    if manifest['intervention_decisions'] != [1,8]:
        raise ValueError('Unexpected intervention horizons')
    result = {'manifest_sha256': canonical_sha(manifest), 'roots': plan['roots'],
              'source_experiment': plan['source_experiment'], 'n_roots': len(plan['roots']),
              'new_branches': len(plan['roots'])*len(manifest['branch_seeds'])*2}
    if (root/'plan.json').exists():
        raise FileExistsError('Plan already frozen')
    atomic_json(root/'plan.json', result)
    print(f"Frozen {result['n_roots']} roots and {result['new_branches']} new branches", flush=True)


def run_branch(manifest, spec, net, cfg, record, trace, oracle, seed, duration, limit=None):
    """Duration 0 observes baseline; 1/8 replace values then resume baseline MCTS."""
    env = make_env(cfg, spec['case']['level'], record)
    search = make_search(cfg, manifest)
    monitor = monitor_at(record, trace, spec['root']['step'], spec['case']['continuation']=='stall_sampled')
    allowance = manifest['max_steps']-spec['root']['step']
    rows, decisions, executed = [], [], []
    try:
        obs = replay_to_branch(env, record, trace, spec['root']['step'])
        lives = int(env.last_info['lives'])
        with numpy_search_rng(seed):
            for step in range(min(limit,allowance) if limit else allowance):
                metadata = monitor.observe(x_position(env.last_info), int(env.last_info['player_state'])) if monitor else {
                    'controller_temperature': 0., 'rescue_trigger': False}
                temperature = metadata['controller_temperature']
                probe = step < (8 if duration == 0 else duration)
                if probe:
                    actual_root = oracle.get(executed)
                    if actual_root['terminal'] or actual_root['observation_sha256'] != obs_sha(obs):
                        raise ValueError('Oracle replay differs from current live observation')
                    before = np.random.get_state()
                    plain = search.run(obs, net, temperature=temperature, deterministic=False)
                    after = np.random.get_state()
                    np.random.set_state(before)
                    lookup = lambda path: oracle.get(tuple(executed)+tuple(path))
                    observed, normal_diagnostic = leaf_search(search, obs, net, temperature, lookup, False)
                    same_search(plain, observed)
                    if any(not np.array_equal(a,b) for a,b in zip(after,np.random.get_state())):
                        raise ValueError('Observation queries changed search RNG')
                    if duration:
                        np.random.set_state(before)
                        result, changed_diagnostic = leaf_search(search, obs, net, temperature, lookup, True)
                    else:
                        result, changed_diagnostic = observed, None
                    decisions.append({'step': step, 'executed_prefix': executed.copy(),
                                      'temperature': temperature, 'normal': normal_diagnostic,
                                      'substituted': changed_diagnostic,
                                      'action_changed': result[0] != plain[0]})
                else:
                    result = search.run(obs, net, temperature=temperature, deterministic=False)
                action, _, q = result
                obs, reward, done, info = env.step(int(action))
                executed.append(int(action))
                new_lives = int(info['lives'])
                rows.append({'action': int(action), 'controller_action': int(action),
                    'temperature': temperature, 'rescue_trigger': metadata['rescue_trigger'],
                    'x_after': x_position(info), 'player_state': int(info['player_state']),
                    'reward': float(reward), 'done': bool(done), 'died': new_lives<lives,
                    'completed': bool(done and info.get('level_complete',False)), 'root_q': float(q)})
                lives = new_lives
                if done:
                    break
    finally:
        env.close()
    return rows, branch_result(seed, duration, decisions,
        summarize_rows(rows,spec['root']['x'],search.discount,manifest['local_horizon'],allowance))


def run_root(root, manifest, plan, root_id, smoke=False):
    import torch
    torch.set_num_threads(1)
    spec = plan['roots'][root_id]
    folder = root/('smoke' if smoke else 'full')/f'root_{root_id:03d}'
    folder.mkdir(parents=True,exist_ok=True)
    identity = {'manifest_sha256': canonical_sha(manifest), 'plan_sha256': canonical_sha(plan),
                'root': spec, 'smoke': smoke, 'script_sha256': file_sha(__file__)}
    identity_sha = canonical_sha(identity)
    if (folder/'identity.json').exists() and json.loads((folder/'identity.json').read_text()) != identity:
        raise ValueError('Resume identity mismatch')
    atomic_json(folder/'identity.json',identity)
    net,cfg = load_model(manifest,spec['case']['level'])
    case = spec['case']
    record,trace = source_data(Path(plan['source_experiment']),case['level'],case['source_condition'],
                              case['episode_index'],manifest['levels'][case['level']]['checkpoint_sha256'])
    cache_path = folder/'observation_values.json'
    saved = None
    if cache_path.exists():
        data=json.loads(cache_path.read_text())
        if data['identity_sha256'] != identity_sha:
            raise ValueError('Cache identity changed')
        saved=data['cache']
    oracle = ObservationValues(cfg,spec,record,trace,net,saved)
    seeds = manifest['branch_seeds'][:1] if smoke else manifest['branch_seeds']
    try:
        with torch.inference_mode():
            for seed in seeds:
                path=folder/f'seed_{seed}.json'
                if path.exists():
                    previous=json.loads(path.read_text())
                    if (previous['identity_sha256'] != identity_sha or not previous['complete'] or
                            previous['trace_sha256'] != file_sha(path.with_suffix('.npz'))):
                        raise ValueError('Saved branch changed')
                    continue
                baseline,old_trace,provenance=old_branch(manifest,spec,'continue',seed)
                rows,audit=run_branch(manifest,spec,net,cfg,record,trace,oracle,seed,0,limit=16)
                check_baseline(rows,old_trace)
                result={'identity_sha256':identity_sha,'seed':seed,'complete':False,
                        'baseline_source':provenance,'baseline_audit':audit,
                        'baseline':{k:baseline[k] for k in ('completed','died','timed_out','discounted_return','local')},
                        'arms':{}}
                arrays={}
                for duration in manifest['intervention_decisions']:
                    rows,branch=run_branch(manifest,spec,net,cfg,record,trace,oracle,seed,duration,
                                          limit=16 if smoke else None)
                    result['arms'][str(duration)]=branch
                    for key in rows[0]:
                        arrays[f'duration_{duration}_{key}']=np.asarray([r[key] for r in rows])
                    print(f"root={root_id} seed={seed} substitute={duration}: complete={branch['completed']} died={branch['died']} cached_paths={len(oracle.cache)}",flush=True)
                atomic_npz(path.with_suffix('.npz'),arrays)
                result.update(complete=True,trace_sha256=file_sha(path.with_suffix('.npz')))
                atomic_json(path,result)
                atomic_json(cache_path,{'identity_sha256':identity_sha,'cache':oracle.export()})
    finally:
        oracle.close()
    atomic_json(folder/'summary.json',{'complete':True,'identity_sha256':identity_sha,'seeds':seeds,
        'branches':len(seeds)*2,'cached_paths':len(oracle.cache)})


def aggregate(root,manifest,plan):
    result={'complete':False,'manifest_sha256':canonical_sha(manifest),'roots':[],'new_branches':0,
            'audit_script_sha256':file_sha(__file__),'audit_job':os.environ.get('SLURM_JOB_ID'),
            'note':'Outcome audit; first-action changes recovered directly from saved traces. Search diagnostics recovery is separate.'}
    for spec in plan['roots']:
        folder=root/'full'/f"root_{spec['root_id']:03d}"
        identity=json.loads((folder/'identity.json').read_text())
        summary=json.loads((folder/'summary.json').read_text())
        if (identity['root'] != spec or identity['manifest_sha256'] != canonical_sha(manifest) or
                identity['smoke'] or not summary['complete'] or summary['seeds'] != manifest['branch_seeds'] or
                summary['identity_sha256'] != canonical_sha(identity)):
            raise ValueError('Root summary incomplete or mismatched')
        groups={'baseline':[], '1':[], '8':[]}
        shifts={'1':0,'8':0}
        for seed in manifest['branch_seeds']:
            path=folder/f'seed_{seed}.json'
            row=json.loads(path.read_text())
            if (not row['complete'] or row['seed'] != seed or row['identity_sha256'] != canonical_sha(identity) or
                    row['trace_sha256'] != file_sha(path.with_suffix('.npz'))):
                raise ValueError('Seed record incomplete or changed')
            old,old_trace,_=old_branch(manifest,spec,'continue',seed)
            if any(row['baseline'][k]!=old[k] for k in row['baseline']):
                raise ValueError('Saved baseline outcome differs from source')
            groups['baseline'].append(row['baseline'])
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as traces:
                for key in ('1','8'):
                    prefix=f'duration_{key}_'
                    actions=traces[prefix+'action']
                    arm=row['arms'][key]
                    if len(actions)!=arm['decisions'] or not len(actions):
                        raise ValueError('Saved length differs from trace')
                    if (bool(np.any(traces[prefix+'completed']))!=arm['completed'] or
                            bool(np.any(traces[prefix+'died']))!=arm['died'] or
                            bool(not traces[prefix+'done'][-1] and len(actions)==manifest['max_steps']-spec['root']['step'])!=arm['timed_out']):
                        raise ValueError('Saved outcome differs from trace')
                    groups[key].append(arm)
                    shifts[key]+=int(actions[0]!=old_trace['action'][0])
                    result['new_branches']+=1
        entry={'root':spec,'groups':{},'first_action_changes':shifts}
        for key,branches in groups.items():
            entry['groups'][key]={'n':len(branches),**{k:sum(b[k] for b in branches)
                for k in ('completed','died','timed_out')}}
            if key!='baseline':
                entry['groups'][key].update(
                    rescued=sum(not b['completed'] and a['completed'] for b,a in zip(groups['baseline'],branches)),
                    lost=sum(b['completed'] and not a['completed'] for b,a in zip(groups['baseline'],branches)))
        result['roots'].append(entry)
    if result['new_branches']!=plan['new_branches']:
        raise ValueError('Branch count mismatch')
    result['complete']=True
    atomic_json(root/'summary.json',result)
    print(f"Audited {result['new_branches']} branches across {len(result['roots'])} roots",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mode',choices=['prepare','run','aggregate'],required=True)
    parser.add_argument('--root',type=int)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Compute nodes only')
    manifest=json.loads((args.out/'manifest.json').read_text())
    if args.mode=='prepare':
        prepare(args.out,manifest)
        return
    plan=json.loads((args.out/'plan.json').read_text())
    if plan['manifest_sha256']!=canonical_sha(manifest):
        raise ValueError('Frozen plan changed')
    if args.mode=='aggregate':
        aggregate(args.out,manifest,plan)
        return
    if args.root is None or not 0<=args.root<len(plan['roots']):
        parser.error('Valid root required')
    import fcntl
    with (args.out/f'.root_{args.root}_{args.smoke}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_root(args.out,manifest,plan,args.root,args.smoke)


if __name__=='__main__':
    main()
