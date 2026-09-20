"""Read-only state-level evidence summary; compute nodes only."""
import argparse
import json
import os
from pathlib import Path
from statistics import mean


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment',type=Path,required=True)
    args=ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        ap.error('Compute nodes only')
    root=args.experiment
    final=json.loads((root/'summary.json').read_text())
    assert final['complete'] and final['finished_branches']==4620
    out={'slurm_job_id':os.environ['SLURM_JOB_ID'],'cases':[], 'note':'Branches and repeated roots are not independent source episodes.'}
    unique_refs=set()
    for c in final['cases']:
        data=json.loads((root/'full'/f"case_{c['case_id']:03d}"/'summary.json').read_text())
        item={k:c[k] for k in ['case_id','level','episode_index','group','source_condition','continuation']}
        item['roots']=[]
        for i,spec in enumerate(c['results']):
            key=(c['level'],c['source_condition'],c['episode_index'],spec['step'])
            if c['group']=='successful_reference':
                if key in unique_refs: continue
                unique_refs.add(key)
            rows=[b for b in data['branches'] if b['root_index']==i]
            reference=[b for b in rows if b['arm'] in ('continue','continue_greedy')]
            prior=reference[0]['prediction']['prior']
            root_info={'step':spec['step'],'x':spec['x'],'arms':spec['arms'],
                       'prior_top':max(range(len(prior)),key=prior.__getitem__),
                       'prior':prior,'selected_actions':sorted(set(b['root_search']['selected_action'] for b in reference)),
                       'top_visit_sets':sorted(set(tuple(j for j,v in enumerate(b['root_search']['visits']) if v==max(b['root_search']['visits'])) for b in reference)),
                       'root_search_return_estimates':reference[0]['root_search']['child_return_estimates'],
                       'predictions':{}}
            for arm in c['arms']:
                a=[b for b in rows if b['arm']==arm]
                root_info['predictions'][arm]={'n':len(a),'observed_discounted_return_mean':mean(b['discounted_return'] for b in a),
                    'predicted_prefix_reward':sum(a[0]['prediction']['prefix_reward_predictions']),
                    'observed_prefix_discounted_reward':a[0]['observed_prefix_discounted_reward'],
                    'imagined_prefix_return':a[0]['prediction']['imagined_prefix_return'],
                    'imagined_endpoint_value':a[0]['prediction']['imagined_endpoint_value'],
                    'observed_endpoint_network_value':a[0]['observed_endpoint_network_value']}
            item['roots'].append(root_info)
        if item['roots']: out['cases'].append(item)
    out['unique_successful_reference_roots']=len(unique_refs)
    (root/'evidence_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    for c in out['cases']:
        print(c['case_id'],c['level'],c['group'],c['episode_index'])
        for r in c['roots']:
            print(' step',r['step'],'x',r['x'],'prior_top',r['prior_top'],'selected',r['selected_actions'],'ties',r['top_visit_sets'])
            print(' outcomes C/D/T', {a:(s['completions'],s['deaths'],s['timeouts']) for a,s in r['arms'].items()})
    print('unique reference roots',len(unique_refs))


if __name__=='__main__':
    main()
