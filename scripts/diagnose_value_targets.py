"""Reconstruct value targets on audited branches; not historical replay targets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_failure_branches import load_model,make_env,source_data
from scripts.diagnose_stall_pulses import replay_to_branch
from scripts.diagnose_value_commitment import old_branch,return_tails
from scripts.eval_controller_diagnostic import atomic_json,canonical_sha,file_sha,observation_uint8,x_position
from src.muzero.targets import build_reanalyze_targets
from src.muzero.returns import compute_n_step_returns


def target_spec(rewards,terminal,position,horizon,discount):
    # Integer observation sentinels recover the exact bootstrap index selected
    # by the production target builder. They are never fed to the network.
    trajectory=SimpleNamespace(length=len(rewards),rewards=np.asarray(rewards,dtype=np.float32),
        terminal=terminal,obs_stacks=np.arange(len(rewards),dtype=np.int64).reshape(-1,1,1,1))
    obs,factor,window=build_reanalyze_targets(trajectory,position,0,horizon,discount)
    return {'reward_window':float(window[0]),'bootstrap_factor':float(factor[0]),
            'bootstrap_index':int(obs[0,0,0,0]) if factor[0] else None}


def target_from_values(spec,values):
    return spec['reward_window']+(spec['bootstrap_factor']*values[spec['bootstrap_index']]
                                  if spec['bootstrap_index'] is not None else 0.)


def replay_and_predict(net,cfg,case,root_spec,record,source_trace,branch_trace,needed):
    import torch
    from src.muzero.transforms import support_to_scalar
    env=make_env(cfg,case['level'],record)
    bank={}
    try:
        obs=replay_to_branch(env,record,source_trace,root_spec['step'])
        lives=int(env.last_info['lives'])
        for t,action in enumerate(branch_trace['action']):
            if t in needed:
                bank[t]=observation_uint8(obs).copy()
            obs,reward,done,info=env.step(int(action))
            new_lives=int(info['lives'])
            if (x_position(info)!=int(branch_trace['x_after'][t]) or
                    int(info['player_state'])!=int(branch_trace['player_state'][t]) or
                    bool(done)!=bool(branch_trace['done'][t]) or
                    bool(new_lives<lives)!=bool(branch_trace['died'][t]) or
                    bool(done and info.get('level_complete',False))!=bool(branch_trace['completed'][t]) or
                    not np.isclose(reward,branch_trace['reward'][t],rtol=0,atol=1e-7)):
                raise ValueError(f'Branch replay mismatch at {t}')
            lives=new_lives
            if done and t+1!=len(branch_trace['action']):
                raise ValueError('Saved actions continue after terminal')
    finally:
        env.close()
    if set(bank)!=set(needed):
        raise ValueError('Missing requested observations')
    support=cfg['model']['value_support']
    values={}
    # Single-observation forwards match the earlier diagnostic numerics.
    with torch.inference_mode():
        for t,obs in sorted(bank.items()):
            tensor=torch.from_numpy(obs).to('cuda',dtype=torch.float32).div_(255).unsqueeze(0)
            _,policy,logits=net.initial_step(tensor)
            value=support_to_scalar(logits.float(),*support)
            probabilities=torch.softmax(logits,-1)[0]
            values[t]={'value':float(value.item()),'logits':logits[0].cpu().tolist(),
                'edge_probability':float((probabilities[0]+probabilities[-1]).item()),
                'policy':torch.softmax(policy,-1)[0].cpu().tolist(),
                'observation_sha256':hashlib.sha256(obs.tobytes()).hexdigest()}
    return values


def support_fit(logits,target,support):
    import torch
    from src.muzero.transforms import cross_entropy_on_support,signed_hyperbolic
    with torch.inference_mode():
        scalar=torch.tensor([target],device='cuda',dtype=torch.float32)
        prediction=torch.tensor([logits],device='cuda',dtype=torch.float32)
        transformed=float(signed_hyperbolic(scalar).item())
        return {'cross_entropy':float(cross_entropy_on_support(prediction,scalar,*support).item()),
                'target_clipped':not support[0]<=transformed<=support[1]}


def prepare(root,manifest):
    previous=Path(manifest['value_experiment'])
    for name,digest in manifest['value_file_sha256'].items():
        if file_sha(previous/name)!=digest:
            raise ValueError(f'Prior experiment changed: {name}')
    original=json.loads((previous/'manifest.json').read_text())
    plan=json.loads((previous/'plan.json').read_text())
    if plan['manifest_sha256']!=canonical_sha(original):
        raise ValueError('Prior plan identity mismatch')
    if not json.loads((previous/'summary.json').read_text())['complete']:
        raise ValueError('Prior branch evidence incomplete')
    for level in manifest['levels']:
        if manifest['levels'][level]['checkpoint_sha256']!=original['levels'][level]['checkpoint_sha256']:
            raise ValueError('Checkpoint identity changed')
    if manifest['branch_seeds']!=original['branch_seeds']:
        raise ValueError('Paired seed list changed')
    if manifest['branch_experiment']!=original['branch_experiment']:
        raise ValueError('Original branch experiment changed')
    current=Path(__file__).resolve().parents[1]
    for path in (previous/'source/src').rglob('*.py'):
        if file_sha(path)!=file_sha(current/path.relative_to(previous/'source')):
            raise ValueError(f'Production implementation changed: {path}')
    result={'manifest_sha256':canonical_sha(manifest),'roots':plan['roots'],
            'source_experiment':plan['source_experiment'],
            'planned_branch_records':len(plan['roots'])*len(manifest['arms'])*len(manifest['branch_seeds'])}
    if (root/'plan.json').exists():
        raise FileExistsError('Plan already frozen')
    atomic_json(root/'plan.json',result)
    print(f"Frozen {len(result['roots'])} roots and {result['planned_branch_records']} existing branch records",flush=True)


def run_root(root,manifest,plan,rid,smoke=False):
    import torch
    torch.set_num_threads(1)
    spec=plan['roots'][rid]; case=spec['case']
    folder=root/('smoke' if smoke else 'full')/f'root_{rid:03d}'
    folder.mkdir(parents=True,exist_ok=True)
    net,cfg=load_model(manifest,case['level'])
    discount=cfg['muzero']['discount']; configured=cfg['muzero']['n_step']
    if not cfg['muzero']['reanalyze'] or configured!=10 or discount!=.999:
        raise ValueError('Frozen training settings differ from protocol')
    buffers={name:tensor.clone() for name,tensor in net.named_buffers()}
    versions={name:parameter._version for name,parameter in net.named_parameters()}
    identity={'manifest_sha256':canonical_sha(manifest),'plan_sha256':canonical_sha(plan),
              'root':spec,'script_sha256':file_sha(__file__),'smoke':smoke,
              'value_support':cfg['model']['value_support'],'configured_n_step':configured,
              'discount':discount,'reanalyze':True,'target_network_saved':False,
              'label':'Current-online bootstrap proxy; not historical lagged targets or original replay.'}
    if (folder/'identity.json').exists() and json.loads((folder/'identity.json').read_text())!=identity:
        raise ValueError('Resume identity mismatch')
    atomic_json(folder/'identity.json',identity)
    record,source_trace=source_data(Path(plan['source_experiment']),case['level'],case['source_condition'],
        case['episode_index'],manifest['levels'][case['level']]['checkpoint_sha256'])
    seeds=manifest['branch_seeds'][:1] if smoke else manifest['branch_seeds']
    cache={}; outputs=[]
    for arm in manifest['arms']:
        for seed in seeds:
            outcome,trace,provenance=old_branch(manifest,spec,arm,seed)
            terminal=bool(trace['done'][-1]); T=len(trace['action'])
            positions=[t for t in manifest['positions'] if t<T]
            specs={(t,h):target_spec(trace['reward'],terminal,t,h,discount)
                   for t in positions for h in manifest['horizons']}
            needed=set(positions)|{s['bootstrap_index'] for s in specs.values() if s['bootstrap_index'] is not None}
            # Exact action/reward/state sequence identity permits emulator inference
            # reuse; each seed retains its own recorded search-Q bootstrap targets.
            sequence=hashlib.sha256()
            for key in ('action','reward','x_after','player_state','done','died','completed'):
                sequence.update(key.encode()); sequence.update(trace[key].tobytes())
            key=sequence.hexdigest()
            if key not in cache:
                cache[key]=replay_and_predict(net,cfg,case,spec['root'],record,source_trace,trace,needed)
            values=cache[key]
            if not needed.issubset(values):
                raise ValueError('Deduplicated trajectory has different required indices')
            scalar_values={t:v['value'] for t,v in values.items()}
            tails=return_tails(trace['reward'],discount)
            # Cross-check the deployed training horizon on every trajectory.
            # Other horizons share this rule and are covered by boundary tests;
            # avoid repeatedly computing all 200-step targets at unused positions.
            stored_targets=compute_n_step_returns(trace['reward'],trace['root_q'],configured,discount,terminal)
            search_values=dict(enumerate(trace['root_q']))
            rows=[]
            for t in positions:
                v=values[t]['value']
                remaining_forced=max(0,len(outcome['forced_actions'])-t)
                row={'position':t,'source_decision':spec['root']['step']+t,
                     'value':v,'observation_sha256':values[t]['observation_sha256'],
                     'edge_probability':values[t]['edge_probability'],
                     'remaining_forced_steps':remaining_forced,
                     'after_prescribed_prefix':remaining_forced==0,
                     'realized_discounted_tail':float(tails[t]),
                     'tail_reaches_true_terminal':terminal,
                     'terminal_mc_fit':support_fit(values[t]['logits'],float(tails[t]),cfg['model']['value_support']) if terminal else None,
                     'targets':{}}
                for horizon in manifest['horizons']:
                    rule=specs[(t,horizon)]
                    td=target_from_values(rule,scalar_values)
                    stored=target_from_values(rule,search_values)
                    if horizon==configured and not np.isclose(stored,stored_targets[t],rtol=1e-5,atol=1e-5):
                        raise ValueError('Production collection and reanalysis target indexing differ')
                    row['targets'][str(horizon)]={**rule,'current_online_bootstrap_target':td,
                        'target_minus_prediction':td-v,'recorded_search_bootstrap_target':float(stored),
                        'bootstrap_value':scalar_values[rule['bootstrap_index']] if rule['bootstrap_index'] is not None else 0.,
                        **support_fit(values[t]['logits'],td,cfg['model']['value_support'])}
                rows.append(row)
            outputs.append({'arm':arm,'seed':seed,'trajectory_key':key,'source':provenance,
                'completed':outcome['completed'],'died':outcome['died'],'timed_out':outcome['timed_out'],
                'terminal':terminal,'rows':rows})
            print(f'root={rid} {arm} seed={seed} terminal={terminal} evaluated_positions={len(rows)} unique_replays={len(cache)}',flush=True)
    if any(not torch.equal(buffers[name],tensor) for name,tensor in net.named_buffers()):
        raise ValueError('Network running buffers changed')
    if any(versions[name]!=parameter._version for name,parameter in net.named_parameters()):
        raise ValueError('Network parameters changed')
    path=folder/'result.json'
    atomic_json(path,{'identity':identity,'complete':True,'branches':outputs,
                     'unique_replayed_trajectories':len(cache),'parameters_and_buffers_unchanged':True})


def aggregate(root,manifest,plan):
    result={'complete':False,'manifest_sha256':canonical_sha(manifest),'roots':[],'branch_records':0}
    for spec in plan['roots']:
        path=root/'full'/f"root_{spec['root_id']:03d}"/'result.json'
        data=json.loads(path.read_text())
        if (not data['complete'] or data['identity']['smoke'] or data['identity']['root']!=spec or
                data['identity']['manifest_sha256']!=canonical_sha(manifest) or
                data['identity']['plan_sha256']!=canonical_sha(plan) or
                data['identity']['script_sha256']!=file_sha(__file__) or not data['parameters_and_buffers_unchanged']):
            raise ValueError('Incomplete or mismatched root')
        actual=[(b['arm'],b['seed']) for b in data['branches']]
        expected=[(a,s) for a in manifest['arms'] for s in manifest['branch_seeds']]
        if actual!=expected:
            raise ValueError('Missing or duplicate branch records')
        result['branch_records']+=len(actual)
        endpoints=[{'arm':b['arm'],'seed':b['seed'],'completed':b['completed'],'timed_out':b['timed_out'],
                    **row} for b in data['branches'] for row in b['rows'] if row['position'] in (0,8)]
        result['roots'].append({'root':spec,'result_sha256':file_sha(path),
            'unique_replays':data['unique_replayed_trajectories'],'endpoints':endpoints})
    if result['branch_records']!=plan['planned_branch_records']:
        raise ValueError('Wrong total')
    result['complete']=True
    atomic_json(root/'summary.json',result)
    print(f"Audited {result['branch_records']} reconstructed branch target records",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['prepare','run','aggregate'],required=True)
    p.add_argument('--root',type=int)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Compute nodes only')
    manifest=json.loads((args.out/'manifest.json').read_text())
    if args.mode=='prepare':
        prepare(args.out,manifest); return
    plan=json.loads((args.out/'plan.json').read_text())
    if plan['manifest_sha256']!=canonical_sha(manifest):
        raise ValueError('Plan changed')
    if args.mode=='aggregate':
        aggregate(args.out,manifest,plan); return
    if args.root is None or not 0<=args.root<len(plan['roots']):
        p.error('Valid root required')
    import fcntl
    with (args.out/f'.root_{args.root}_{args.smoke}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_root(args.out,manifest,plan,args.root,args.smoke)


if __name__=='__main__':
    main()
