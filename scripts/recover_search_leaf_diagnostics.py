"""Recover search metadata without rerunning complete experimental trajectories."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_search_leaf_values import run_branch
from scripts.diagnose_failure_branches import load_model, source_data
from scripts.diagnose_value_commitment import old_branch, check_baseline
from scripts.eval_controller_diagnostic import atomic_json, canonical_sha, file_sha


class CachedValues:
    def __init__(self, rows):
        self.data={tuple(row['actions']):row['result'] for row in rows}

    def get(self, actions):
        key=tuple(actions)
        if key not in self.data:
            raise ValueError(f'Recovery requested an unrecorded observation path: {key}')
        return self.data[key]


def recover(root, rid):
    import torch
    torch.set_num_threads(1)
    manifest=json.loads((root/'manifest.json').read_text())
    plan=json.loads((root/'plan.json').read_text())
    if plan['manifest_sha256']!=canonical_sha(manifest):
        raise ValueError('Plan changed')
    spec=plan['roots'][rid]
    directory=root/'full'/f'root_{rid:03d}'
    identity=json.loads((directory/'identity.json').read_text())
    if identity['root']!=spec or identity['manifest_sha256']!=canonical_sha(manifest) or identity['smoke']:
        raise ValueError('Root identity changed')
    cache=json.loads((directory/'observation_values.json').read_text())
    if cache['identity_sha256']!=canonical_sha(identity):
        raise ValueError('Observation cache identity changed')
    # Compare every production source file; only the diagnostic serializer changed.
    current=Path(__file__).resolve().parents[1]
    for path in (root/'source/src').rglob('*.py'):
        if file_sha(path)!=file_sha(current/path.relative_to(root/'source')):
            raise ValueError(f'Production implementation changed: {path}')
    net,cfg=load_model(manifest,spec['case']['level'])
    case=spec['case']
    record,trace=source_data(Path(plan['source_experiment']),case['level'],case['source_condition'],
        case['episode_index'],manifest['levels'][case['level']]['checkpoint_sha256'])
    oracle=CachedValues(cache['cache'])
    result={'complete':False,'identity_sha256':canonical_sha(identity),
            'recovery_script_sha256':file_sha(__file__),
            'runner_sha256':file_sha(current/'scripts/diagnose_search_leaf_values.py'),
            'cache_sha256':file_sha(directory/'observation_values.json'),
            'slurm_job_id':os.environ['SLURM_JOB_ID'],'seeds':[]}
    with torch.inference_mode():
        for seed in manifest['branch_seeds']:
            path=directory/f'seed_{seed}.json'
            saved=json.loads(path.read_text())
            if (saved['identity_sha256']!=canonical_sha(identity) or not saved['complete'] or
                    saved['trace_sha256']!=file_sha(path.with_suffix('.npz'))):
                raise ValueError('Original branch changed')
            _,baseline,_=old_branch(manifest,spec,'continue',seed)
            rows,details=run_branch(manifest,spec,net,cfg,record,trace,oracle,seed,0,limit=16)
            check_baseline(rows,baseline)
            item={'seed':seed,'original_record_sha256':file_sha(path),
                  'baseline':details['search_decisions'],'arms':{}}
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as archive:
                for duration in (1,8):
                    rows,details=run_branch(manifest,spec,net,cfg,record,trace,oracle,seed,duration,limit=8)
                    prefix=f'duration_{duration}_'
                    expected={key[len(prefix):]:archive[key] for key in archive.files if key.startswith(prefix)}
                    check_baseline(rows,expected)
                    item['arms'][str(duration)]=details['search_decisions']
            result['seeds'].append(item)
            print(f'root={rid} seed={seed}: original actions, rewards, states, Q and cached observations match',flush=True)
    result['complete']=True
    atomic_json(directory/'search_recovery.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--root',type=int,required=True)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Compute nodes only')
    recover(args.out,args.root)
