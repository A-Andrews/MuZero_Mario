"""Separate imagined-value inconsistency from interruption of selected actions.

Frozen conditional probes, not new independent level-performance evaluations.
Reuse audited full continuations; generate new first-action commitment branches.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_failure_branches import (load_model, make_env, make_search,
    monitor_at, observed_search, source_data, summarize_rows)
from scripts.diagnose_stall_pulses import replay_to_branch
from scripts.eval_controller_diagnostic import (atomic_json, atomic_npz, canonical_sha,
    file_sha, numpy_search_rng, observation_uint8, x_position)


def unique_roots(cases):
    """Deduplicate repeated successful references, never distinct source episodes."""
    roots = {}
    for case in cases:
        if case['group'] == 'lost_success_trigger_comparison':
            continue
        for index, root in enumerate(case['roots']):
            key = (case['level'], case['source_condition'], case['episode_index'], root['step'])
            reference = {'case_id': case['case_id'], 'root_index': index}
            if key in roots:
                if roots[key]['case']['group'] != case['group']:
                    raise ValueError('Root group differs across references')
                roots[key]['aliases'].append(reference)
            else:
                roots[key] = {'root_id': len(roots), 'case': case, 'root': root,
                              'reference': reference, 'aliases': [reference]}
    return list(roots.values())


def return_tails(rewards, discount):
    tails = np.zeros(len(rewards) + 1, dtype=np.float64)
    for i in range(len(rewards)-1, -1, -1):
        tails[i] = float(rewards[i]) + discount * tails[i+1]
    return tails


def decompose_error(imagined_rewards, real_rewards, imagined_value, observed_value,
                    realized_tail, discount):
    if len(imagined_rewards) != len(real_rewards):
        raise ValueError('Prediction and reality must use the same horizon')
    h = len(real_rewards)
    powers = discount ** np.arange(h)
    predicted_prefix = float(np.dot(powers, imagined_rewards))
    real_prefix = float(np.dot(powers, real_rewards))
    estimate = predicted_prefix + discount**h * imagined_value
    realized = real_prefix + discount**h * realized_tail
    parts = {'reward_error': predicted_prefix-real_prefix,
             'imagined_state_value_gap': discount**h * (imagined_value-observed_value),
             'observed_state_value_residual': discount**h * (observed_value-realized_tail)}
    if not np.isclose(sum(parts.values()), estimate-realized, atol=1e-8):
        raise ValueError('Return error decomposition failed')
    return {**parts, 'imagined_return': estimate, 'realized_return': realized,
            'total_error': estimate-realized,
            'actual_rewards_imagined_value': real_prefix+discount**h*imagined_value,
            'actual_rewards_observed_value': real_prefix+discount**h*observed_value}


def held_action(proposed, first_action, step, duration):
    return int(first_action if step < duration else proposed)


def old_branch(manifest, spec, arm, seed):
    ref = spec['reference']
    directory = Path(manifest['branch_experiment'])/'full'/f"case_{ref['case_id']:03d}"
    path = directory/f"root_{ref['root_index']}_seed_{seed}_{arm}.json"
    result = json.loads(path.read_text())
    identity = json.loads((directory/'identity.json').read_text())
    if (result['identity_sha256'] != canonical_sha(identity) or
            identity['case'] != spec['case'] or result['seed'] != seed or
            result['arm'] != arm or result['step'] != spec['root']['step'] or
            result['root_index'] != ref['root_index'] or
            file_sha(path.with_suffix('.npz')) != result['trace_sha256']):
        raise ValueError(f'Prior branch provenance mismatch: {path}')
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as archive:
        trace = {k: archive[k] for k in archive.files}
    return result, trace, {'path': str(path), 'sha256': file_sha(path),
                           'trace_sha256': result['trace_sha256']}


def prepare(root, manifest):
    prior = Path(manifest['branch_experiment'])
    for name, digest in manifest['branch_file_sha256'].items():
        if file_sha(prior/name) != digest:
            raise ValueError(f'Prior experiment changed: {name}')
    original = json.loads((prior/'manifest.json').read_text())
    original_plan = json.loads((prior/'branch_plan.json').read_text())
    if not json.loads((prior/'summary.json').read_text())['complete']:
        raise ValueError('Prior experiment incomplete')
    if original_plan['manifest_sha256'] != canonical_sha(original):
        raise ValueError('Prior plan identity mismatch')
    for key in ('branch_seeds', 'max_steps', 'num_simulations', 'leaf_batch', 'root_exploration_eps'):
        if manifest[key] != original[key]:
            raise ValueError(f'Paired settings changed: {key}')
    for level, settings in manifest['levels'].items():
        if settings['checkpoint_sha256'] != original['levels'][level]['checkpoint_sha256']:
            raise ValueError('Paired checkpoint changed')
    # Reusing outcomes requires unchanged environment, search and network code.
    current = Path(__file__).resolve().parents[1]
    paths = list((prior/'source/src').rglob('*.py'))
    paths += [prior/'source/scripts'/name for name in (
        'diagnose_failure_branches.py', 'diagnose_stall_pulses.py',
        'eval_controller_diagnostic.py', 'stall_sampling_controller.py')]
    for path in paths:
        if file_sha(path) != file_sha(current/path.relative_to(prior/'source')):
            raise ValueError(f'Reused baseline implementation changed: {path}')
    roots = unique_roots(original_plan['cases'])
    plan = {'manifest_sha256': canonical_sha(manifest), 'roots': roots,
            'source_experiment': original['source_experiment'],
            'n_unique_roots': len(roots),
            'n_source_episodes': len({(r['case']['level'], r['case']['source_condition'],
                                    r['case']['episode_index']) for r in roots}),
            'new_commitment_branches': len(roots)*3*len(manifest['branch_seeds'])}
    if (root/'plan.json').exists():
        raise FileExistsError('Plan is already frozen')
    atomic_json(root/'plan.json', plan)
    print(f"Frozen {len(roots)} roots, {plan['new_commitment_branches']} new branches", flush=True)


def initial(net, obs):
    import torch
    return net.initial_inference(torch.from_numpy(obs).to('cuda', dtype=torch.float32).unsqueeze(0))


def predict_values(manifest, spec, net, cfg, record, source_trace):
    """Replay identical prefixes; evaluate imagined, reanchored and observed values."""
    import torch
    case, t = spec['case'], spec['root']['step']
    discount = cfg['muzero']['discount']
    report = {}
    for arm in manifest['value_arms']:
        old = [old_branch(manifest, spec, arm, seed) for seed in manifest['branch_seeds']]
        prefix = old[0][0]['forced_actions']
        if len(prefix) != manifest['prefix_decisions']:
            raise ValueError('Unexpected prefix length')
        horizon = min(len(prefix), len(old[0][1]['action']))
        for result, trace, _ in old:
            if result['forced_actions'] != prefix:
                raise ValueError('Forced actions changed between seeds')
            for key in ('action', 'reward', 'x_after', 'done', 'player_state'):
                if not np.array_equal(trace[key][:horizon], old[0][1][key][:horizon]):
                    raise ValueError('Identical prefixes reached different states')
        env = make_env(cfg, case['level'], record)
        rows, real_rewards, predicted_rewards = [], [], []
        tails = [return_tails(trace['reward'], discount) for _, trace, _ in old]
        try:
            obs = replay_to_branch(env, record, source_trace, t)
            imagined, _, _ = initial(net, obs)
            observed = imagined
            for i, action in enumerate(prefix[:horizon]):
                a = torch.tensor([action], device='cuda')
                imagined, predicted_reward, logits, value = net.recurrent_inference(imagined, a)
                _, anchored_reward, anchored_logits, anchored_value = net.recurrent_inference(observed, a)
                obs, reward, done, info = env.step(action)
                expected = old[0][1]
                if (x_position(info) != int(expected['x_after'][i]) or
                        bool(done) != bool(expected['done'][i]) or
                        int(info['player_state']) != int(expected['player_state'][i]) or
                        not np.isclose(reward, expected['reward'][i], atol=1e-7, rtol=0)):
                    raise ValueError('Value probe diverged from saved prefix')
                observed, observed_logits, observed_value = initial(net, obs)
                v_real = 0. if done else float(observed_value.item())
                real_rewards.append(float(reward))
                predicted_rewards.append(float(predicted_reward.item()))
                h = i+1
                row = {'horizon': h, 'action': int(action), 'done': bool(done),
                       'observation_sha256': __import__('hashlib').sha256(observation_uint8(obs).tobytes()).hexdigest(),
                       'imagined_reward': float(predicted_reward.item()), 'real_reward': float(reward),
                       'reanchored_one_step_reward': float(anchored_reward.item()),
                       'imagined_value': float(value.item()), 'observed_value': v_real,
                       'observed_value_raw': float(observed_value.item()),
                       'reanchored_one_step_value': float(anchored_value.item()),
                       'imagined_policy': torch.softmax(logits, -1)[0].cpu().tolist(),
                       'observed_policy': torch.softmax(observed_logits, -1)[0].cpu().tolist(),
                       'reanchored_one_step_policy': torch.softmax(anchored_logits, -1)[0].cpu().tolist(),
                       'by_seed': []}
                for seed, tail in zip(manifest['branch_seeds'], tails):
                    row['by_seed'].append({'seed': seed, 'realized_tail': float(tail[h]),
                        **decompose_error(predicted_rewards, real_rewards, float(value.item()),
                                          v_real, float(tail[h]), discount)})
                rows.append(row)
                if done:
                    break
        finally:
            env.close()
        report[arm] = {'forced_actions': prefix, 'steps': rows,
                       'source_branches': [source for _, _, source in old]}
    return report


def rollout(manifest, spec, net, cfg, record, source_trace, seed, duration, limit=None):
    case, t = spec['case'], spec['root']['step']
    env = make_env(cfg, case['level'], record)
    search = make_search(cfg, manifest)
    monitor = monitor_at(record, source_trace, t, case['continuation'] == 'stall_sampled')
    allowance = manifest['max_steps']-t
    rows, diagnostics = [], []
    try:
        obs = replay_to_branch(env, record, source_trace, t)
        lives = int(env.last_info['lives'])
        with numpy_search_rng(seed):
            for step in range(min(allowance, limit) if limit else allowance):
                metadata = monitor.observe(x_position(env.last_info), int(env.last_info['player_state'])) if monitor else {
                    'controller_temperature': 0., 'rescue_trigger': False}
                temperature = metadata['controller_temperature']
                if step < 16:
                    (proposed, visits, q), diagnostic = observed_search(search, obs, net, temperature)
                    diagnostics.append({'step': step, 'proposed_action': int(proposed),
                                        'visits': visits.tolist(), **diagnostic})
                else:
                    proposed, _, q = search.run(obs, net, temperature=temperature, deterministic=False)
                if step == 0:
                    first_action = int(proposed)
                action = held_action(proposed, first_action, step, duration)
                obs, reward, done, info = env.step(action)
                new_lives = int(info['lives'])
                rows.append({'action': action, 'controller_action': int(proposed),
                    'temperature': temperature, 'rescue_trigger': metadata['rescue_trigger'],
                    'forced': step < duration, 'x_after': x_position(info),
                    'player_state': int(info['player_state']), 'reward': float(reward),
                    'done': bool(done), 'died': new_lives < lives,
                    'completed': bool(done and info.get('level_complete', False)), 'root_q': float(q)})
                lives = new_lives
                if done:
                    break
    finally:
        env.close()
    return rows, {'seed': seed, 'duration': duration, 'first_action': first_action,
                  'first_shadow_disagreement': next((i for i, row in enumerate(rows[:duration])
                        if row['action'] != row['controller_action']), None),
                  'search_first_16': diagnostics,
                  **summarize_rows(rows, spec['root']['x'], search.discount,
                                   manifest['local_horizon'], allowance)}


def check_baseline(rows, trace):
    for key in ('action', 'controller_action', 'temperature', 'rescue_trigger', 'x_after',
                'player_state', 'reward', 'done', 'died', 'completed', 'root_q'):
        actual = np.asarray([r[key] for r in rows])
        expected = trace[key][:len(rows)]
        if not np.allclose(actual, expected, rtol=0, atol=1e-5 if key == 'root_q' else 1e-7):
            raise ValueError(f'Paired baseline mismatch: {key}')


def run_root(root, manifest, plan, root_id, mode, smoke=False):
    import torch
    torch.set_num_threads(1)
    spec = plan['roots'][root_id]
    case = spec['case']
    folder = root/('smoke' if smoke else 'full')/f'root_{root_id:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    identity = {'manifest_sha256': canonical_sha(manifest), 'plan_sha256': canonical_sha(plan),
                'root': spec, 'script_sha256': file_sha(__file__), 'smoke': smoke}
    path = folder/'identity.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError('Resume identity mismatch')
    atomic_json(path, identity)
    net, cfg = load_model(manifest, case['level'])
    record, source_trace = source_data(Path(plan['source_experiment']), case['level'],
        case['source_condition'], case['episode_index'], manifest['levels'][case['level']]['checkpoint_sha256'])
    seeds = manifest['branch_seeds'][:1] if smoke else manifest['branch_seeds']
    with torch.inference_mode():
        if mode in ('values', 'both'):
            values = predict_values(manifest, spec, net, cfg, record, source_trace)
            atomic_json(folder/'values.json', {'identity_sha256': canonical_sha(identity),
                                             'complete': True, 'arms': values})
            print(f'root={root_id} values complete', flush=True)
        if mode not in ('commitment', 'both'):
            return
        baselines = []
        for seed in seeds:
            result, trace, provenance = old_branch(manifest, spec, 'continue', seed)
            rows, _ = rollout(manifest, spec, net, cfg, record, source_trace, seed, 1,
                             limit=manifest['baseline_replay_decisions'])
            check_baseline(rows, trace)
            baselines.append({'seed': seed, 'source': provenance, 'replayed_decisions': len(rows),
                              'first_action': int(trace['action'][0]),
                              'first_action_switch': next((i for i, a in enumerate(trace['action'][:16])
                                                          if a != trace['action'][0]), None),
                              'duration': 1,
                              **{k: result[k] for k in ('completed', 'died', 'timed_out', 'discounted_return', 'local')}})
        atomic_json(folder/'baseline.json', {'identity_sha256': canonical_sha(identity),
                    'branches': baselines, 'complete': True})
        for duration in manifest['hold_decisions'][1:]:
            path = folder/f'hold_{duration}.json'
            if path.exists():
                saved = json.loads(path.read_text())
                if (saved['identity_sha256'] != canonical_sha(identity) or not saved['complete'] or
                        saved['trace_sha256'] != file_sha(path.with_suffix('.npz')) or
                        [b['seed'] for b in saved['branches']] != seeds):
                    raise ValueError('Saved commitment group changed')
                continue
            branches, arrays = [], {}
            for seed, baseline in zip(seeds, baselines):
                rows, branch = rollout(manifest, spec, net, cfg, record, source_trace,
                                       seed, duration, limit=16 if smoke else None)
                if branch['first_action'] != baseline['first_action']:
                    raise ValueError('Paired first action changed')
                branches.append(branch)
                for key in rows[0]:
                    arrays[f'seed_{seed}_{key}'] = np.asarray([r[key] for r in rows])
                print(f"root={root_id} hold={duration} seed={seed} complete={branch['completed']} death={branch['died']}", flush=True)
            atomic_npz(path.with_suffix('.npz'), arrays)
            atomic_json(path, {'identity_sha256': canonical_sha(identity), 'complete': True,
                              'branches': branches, 'trace_sha256': file_sha(path.with_suffix('.npz'))})


def aggregate(root, manifest, plan):
    report = {'complete': True, 'manifest_sha256': canonical_sha(manifest), 'roots': [],
              'n_unique_roots': plan['n_unique_roots'], 'new_branches': 0,
              'note': 'Conditional branch repetitions; report episodes and roots separately. No population completion-rate claim.'}
    for spec in plan['roots']:
        folder = root/'full'/f"root_{spec['root_id']:03d}"
        identity = json.loads((folder/'identity.json').read_text())
        if identity['root'] != spec or identity['manifest_sha256'] != canonical_sha(manifest):
            raise ValueError('Root identity changed')
        entry = {'root': spec, 'holds': {}, 'value_horizon_means': {}}
        for name, duration in [('baseline', 1), ('hold_2', 2), ('hold_4', 4), ('hold_8', 8)]:
            data = json.loads((folder/f'{name}.json').read_text())
            if (not data['complete'] or data['identity_sha256'] != canonical_sha(identity) or
                    [b['seed'] for b in data['branches']] != manifest['branch_seeds']):
                raise ValueError('Incomplete or changed commitment group')
            if duration != 1:
                if data['trace_sha256'] != file_sha(folder/f'{name}.npz'):
                    raise ValueError('Changed commitment trace')
                report['new_branches'] += len(data['branches'])
            else:
                for branch in data['branches']:
                    old_branch(manifest, spec, 'continue', branch['seed'])
            entry['holds'][str(duration)] = {k: sum(b[k] for b in data['branches'])
                                            for k in ('completed', 'died', 'timed_out')}
            entry['holds'][str(duration)]['n'] = len(data['branches'])
        values = json.loads((folder/'values.json').read_text())
        if not values['complete'] or values['identity_sha256'] != canonical_sha(identity):
            raise ValueError('Invalid value probe')
        for arm, data in values['arms'].items():
            entry['value_horizon_means'][arm] = [{
                'horizon': row['horizon'], 'imagined_value': row['imagined_value'],
                'observed_value': row['observed_value'],
                'reanchored_one_step_value': row['reanchored_one_step_value'],
                **{key: float(np.mean([sample[key] for sample in row['by_seed']])) for key in (
                    'reward_error', 'imagined_state_value_gap', 'observed_state_value_residual', 'total_error')}
            } for row in data['steps']]
        report['roots'].append(entry)
    if report['new_branches'] != plan['new_commitment_branches']:
        raise ValueError('Wrong branch count')
    atomic_json(root/'summary.json', report)
    print(f"Audited {report['new_branches']} new branches and {len(report['roots'])} value roots", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=['prepare', 'values', 'commitment', 'both', 'aggregate'], required=True)
    parser.add_argument('--root', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Compute nodes only')
    manifest = json.loads((args.out/'manifest.json').read_text())
    if args.mode == 'prepare':
        prepare(args.out, manifest)
        return
    plan = json.loads((args.out/'plan.json').read_text())
    if plan['manifest_sha256'] != canonical_sha(manifest):
        raise ValueError('Frozen plan changed')
    if args.mode == 'aggregate':
        aggregate(args.out, manifest, plan)
        return
    if args.root is None or not 0 <= args.root < len(plan['roots']):
        parser.error('Valid root required')
    import fcntl
    with (args.out/f'.root_{args.root}_{args.smoke}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_root(args.out, manifest, plan, args.root, args.mode, args.smoke)


if __name__ == '__main__':
    main()
