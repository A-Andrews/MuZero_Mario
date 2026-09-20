"""Summarize audited paired leaf-value interventions and any recovered searches."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.summarize_value_commitment import signature
from scripts.eval_controller_diagnostic import atomic_json, file_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--require-recovery',action='store_true')
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Compute nodes only')
    report=json.loads((args.out/'summary.json').read_text())
    if not report['complete'] or report['new_branches']!=1120:
        raise ValueError('Outcome audit incomplete')
    evidence={'outcome_audit_sha256':file_sha(args.out/'summary.json'),'groups':[],
              'script_sha256':file_sha(__file__),'job_id':os.environ['SLURM_JOB_ID'],
              'roots':[],'recovered_roots':0}
    groups=defaultdict(list)
    for item in report['roots']:
        spec=item['root']; case=spec['case']; rid=spec['root_id']
        row={'root_id':rid,'level':case['level'],'condition':case['source_condition'],
             'episode':case['episode_index'],'step':spec['root']['step'],
             'signature':signature(spec),'outcomes':item['groups'],
             'first_action_changes':item['first_action_changes']}
        recovery=args.out/'full'/f'root_{rid:03d}'/'search_recovery.json'
        if recovery.exists():
            searches=json.loads(recovery.read_text())
            if not searches['complete'] or len(searches['seeds'])!=10:
                raise ValueError('Partial search recovery')
            evidence['recovered_roots']+=1
            row['recovery_sha256']=file_sha(recovery)
            row['search']={}
            for arm in ('baseline','1','8'):
                summaries=[]; pairs=Counter(); depth=Counter(); gaps=[]; terminals=0; shifts=0
                for seed in searches['seeds']:
                    decisions=seed['baseline'] if arm=='baseline' else seed['arms'][arm]
                    for decision in decisions:
                        normal=decision['normal']; actual=normal if arm=='baseline' else decision['substituted']
                        pairs[f"{normal['action']}->{actual['action']}"]+=1
                        shifts+=int(decision['action_changed'])
                        summaries.append(actual['max_depth'])
                        for event in actual['leaf_backups']:
                            depth[event['depth']]+=1
                            gaps.append(abs(event['imagined_value']-event['observed_value']))
                            terminals+=int(event['terminal'])
                row['search'][arm]={'n_decisions':len(summaries),'action_changes':shifts,
                    'action_pairs':dict(pairs),'backup_depths':dict(depth),
                    'maximum_depth':max(summaries),'mean_abs_value_gap':float(np.mean(gaps)),
                    'terminal_backups':terminals,'n_backups':len(gaps)}
        elif args.require_recovery:
            raise ValueError(f'Missing search recovery: {rid}')
        evidence['roots'].append(row)
        timing=str(spec['root']['step']) if case['level']=='Level6-1' and case['group']=='early_death' else 'all'
        groups[(row['level'],row['condition'],row['signature'],timing)].append(row)
        print(rid,row['level'],'ep',row['episode'],'t',row['step'],row['signature'],
              'outcomes',json.dumps(row['outcomes']),'action_changes',row['first_action_changes'])
    print('\nGROUPS: conditional branches, not independent performance attempts')
    for (level,condition,group,timing),rows in groups.items():
        item={'level':level,'condition':condition,'signature':group,'timing':timing,
              'n_episodes':len({r['episode'] for r in rows}),'n_roots':len(rows),'outcomes':{},
              'first_action_changes':{arm:sum(r['first_action_changes'][arm] for r in rows) for arm in ('1','8')}}
        for arm in ('baseline','1','8'):
            item['outcomes'][arm]={key:sum(r['outcomes'][arm][key] for r in rows) for key in rows[0]['outcomes'][arm]}
        evidence['groups'].append(item)
        print(json.dumps(item))
    print('\nRecovered search diagnostics:',evidence['recovered_roots'],'/ 56 roots')
    for row in evidence['roots']:
        if 'search' in row:
            print('SEARCH',row['root_id'],json.dumps(row['search']))
    evidence['search_recovery_complete']=evidence['recovered_roots']==56
    atomic_json(args.out/'evidence_summary.json',evidence)


if __name__=='__main__':
    main()
