"""Read-only result analysis of the frozen value/commitment diagnostics."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(spec):
    case, x = spec['case'], spec['root']['x']
    if case['group'] != 'early_death':
        return case['group']
    if case['level'] == 'Level6-1':
        return 'obstacle_1394' if x == 1394 else 'pit_2807'
    return 'pipe_898' if x == 898 else 'later_death_1948'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Compute nodes only')
    root = args.out
    summary = json.loads((root/'summary.json').read_text())
    if not summary['complete'] or summary['new_branches'] != 1680:
        raise ValueError('Incomplete experiment')
    report = {'slurm_job_id': os.environ['SLURM_JOB_ID'],
              'script_sha256': digest(Path(__file__)), 'summary_sha256': digest(root/'summary.json'),
              'roots': [], 'groups': [], 'value_error_by_level_and_group': []}
    grouped = defaultdict(list)
    errors = defaultdict(list)
    print('ROOT RESULTS: completions/deaths/timeouts for hold 1,2,4,8; each denominator=10')
    for entry in summary['roots']:
        spec = entry['root']; case = spec['case']; rid = spec['root_id']
        folder = root/'full'/f'root_{rid:03d}'
        baseline = json.loads((folder/'baseline.json').read_text())['branches']
        paired = {}
        first = Counter(b['first_action'] for b in baseline)
        for duration in (2, 4, 8):
            branches = json.loads((folder/f'hold_{duration}.json').read_text())['branches']
            by_seed = {b['seed']: b for b in branches}
            paired[str(duration)] = {
                'rescued': sum(not b['completed'] and by_seed[b['seed']]['completed'] for b in baseline),
                'lost': sum(b['completed'] and not by_seed[b['seed']]['completed'] for b in baseline),
                'shadow_disagreements': [b['first_shadow_disagreement'] for b in branches],
            }
        values = json.loads((folder/'values.json').read_text())
        value_rows = {}
        for arm, data in values['arms'].items():
            value_rows[arm] = []
            for step in data['steps']:
                v, actual, anchored = (step[k] for k in ('imagined_value', 'observed_value', 'reanchored_one_step_value'))
                row = {'h': step['horizon'], 'imagined': v, 'observed': actual, 'reanchored': anchored,
                       'imagined_abs_gap': abs(v-actual), 'reanchored_abs_gap': abs(anchored-actual)}
                for k in ('reward_error', 'imagined_state_value_gap', 'observed_state_value_residual',
                          'total_error', 'imagined_return', 'realized_return', 'actual_rewards_observed_value'):
                    row[k] = float(np.mean([r[k] for r in step['by_seed']]))
                value_rows[arm].append(row)
                errors[(case['level'], signature(spec), step['horizon'])].append(row)
        item = {'root_id': rid, 'level': case['level'], 'episode': case['episode_index'],
                'condition': case['source_condition'], 'signature': signature(spec),
                'step': spec['root']['step'], 'x': spec['root']['x'],
                'first_actions': dict(first), 'first_switches': [b['first_action_switch'] for b in baseline],
                'holds': entry['holds'], 'paired': paired, 'values': value_rows,
                'value_file_sha256': digest(folder/'values.json')}
        if rid in (0,1,35,37,39,40,41):
            saved = Path(baseline[0]['source']['path']).with_suffix('.npz')
            if digest(saved) != baseline[0]['source']['trace_sha256']:
                raise ValueError('Saved baseline trace changed')
            with np.load(saved, allow_pickle=False) as trace:
                item['example_baseline_actions_first_16'] = trace['action'][:16].tolist()
            item['example_hold8_search_proposals'] = [d['proposed_action'] for d in
                json.loads((folder/'hold_8.json').read_text())['branches'][0]['search_first_16']]
        report['roots'].append(item)
        grouped[(item['level'], item['condition'], item['signature'])].append(item)
        outcomes = ['/'.join(str(entry['holds'][str(h)][k]) for k in ('completed','died','timed_out')) for h in (1,2,4,8)]
        print(f"{rid:2} {case['level']} ep={case['episode_index']:2} {item['signature']:22} t={item['step']:3} x={item['x']:4} first={dict(first)} {'  '.join(outcomes)}")
    print('\nGROUPS (diagnostic branches, not independent episode performance)')
    for (level, condition, group), rows in grouped.items():
        item = {'level': level, 'condition': condition, 'signature': group,
                'n_episodes': len({r['episode'] for r in rows}), 'n_roots': len(rows), 'holds': {}}
        for h in (1,2,4,8):
            item['holds'][str(h)] = {k: sum(r['holds'][str(h)][k] for r in rows)
                                     for k in ('n','completed','died','timed_out')}
            if h != 1:
                item['holds'][str(h)].update({k: sum(r['paired'][str(h)][k] for r in rows) for k in ('rescued','lost')})
        report['groups'].append(item)
        print(json.dumps(item))
    print('\nVALUE GAPS: means across roots and four prefixes, never treating seeds as new predictions')
    for (level, group, horizon), rows in errors.items():
        item = {'level': level, 'group': group, 'horizon': horizon, 'n_root_prefixes': len(rows),
                **{k: float(np.mean([r[k] for r in rows])) for k in ('imagined_abs_gap','reanchored_abs_gap')},
                'reanchoring_improves_n': sum(r['reanchored_abs_gap'] < r['imagined_abs_gap']-1e-5 for r in rows)}
        report['value_error_by_level_and_group'].append(item)
        if horizon in (1,4,8):
            print(json.dumps(item))
    print('\nEXAMPLE EIGHT-STEP VALUE DECOMPOSITIONS')
    for rid in (0,1,2,8,9,35,36,37,38,39,40,41):
        row = report['roots'][rid]
        print(f"root={rid} {row['level']} ep={row['episode']} t={row['step']}")
        for arm, values in row['values'].items():
            print(arm, json.dumps({k: round(v,2) for k,v in values[-1].items()}))
    print('\nEXAMPLE ACTION SEQUENCES (first seed, descriptive examples)')
    for row in report['roots']:
        if 'example_baseline_actions_first_16' in row:
            print(row['root_id'], 'baseline', row['example_baseline_actions_first_16'],
                  'hold8 shadow', row['example_hold8_search_proposals'])
    (root/'evidence_summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
