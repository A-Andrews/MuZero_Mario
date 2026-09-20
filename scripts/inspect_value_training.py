"""Inspect frozen value-training settings and finish search-recovery accounting."""
import json
import os
from pathlib import Path
import sys
from collections import Counter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.checkpoint import load_checkpoint
from scripts.eval_controller_diagnostic import atomic_json, file_sha


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Compute nodes only')
    source=Path('outputs/controller_policy/value-commitment-v1-20260918')
    manifest=json.loads((source/'manifest.json').read_text())
    result={'job_id':os.environ['SLURM_JOB_ID'],'levels':{},'search':{}}
    for level,settings in manifest['levels'].items():
        path=Path(settings['checkpoint'])
        if file_sha(path)!=settings['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint changed')
        checkpoint=load_checkpoint(path,map_location='cpu')
        cfg=checkpoint['cfg_snapshot']
        run=Path('outputs/runs')/('spec-'+level.lower())
        result['levels'][level]={'checkpoint':str(path),'sha256':settings['checkpoint_sha256'],
            'payload_keys':sorted(checkpoint),'training_step':checkpoint['training_step'],
            'env_step':checkpoint['env_step'],'muzero':cfg['muzero'],
            'training':{k:v for k,v in cfg['training'].items() if k in (
                'target_update_every','batch_size','learning_rate','lr','max_train_per_env_step')},
            'selfplay':cfg['selfplay'],'run_directory_entries':sorted(p.name for p in run.iterdir()) if run.exists() else []}
        print(level,json.dumps(result['levels'][level]),flush=True)
    evidence=json.loads(Path('outputs/controller_policy/search-leaf-values-v1b-20260919/evidence_summary.json').read_text())
    if not evidence['search_recovery_complete'] or evidence['recovered_roots']!=56:
        raise ValueError('Search recovery incomplete')
    for arm in ('baseline','1','8'):
        depths=Counter(); terminals=0; decisions=0
        for row in evidence['roots']:
            data=row['search'][arm]
            depths.update({int(k):v for k,v in data['backup_depths'].items()})
            terminals+=data['terminal_backups']; decisions+=data['n_decisions']
        result['search'][arm]={'depth_counts':dict(sorted(depths.items())),
            'terminal_backups':terminals,'total_backups':sum(depths.values()),'decisions':decisions}
    print('SEARCH',json.dumps(result['search']),flush=True)
    out=Path('outputs/controller_policy/value-target-audit-v1-20260920')
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'preflight.json',result)


if __name__=='__main__':
    main()
