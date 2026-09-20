"""Render a slide from saved distributions, with equal weights across levels."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('images/action_policy_search/distributions.json'))
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Render on a compute node')
    raw = args.source.read_bytes()
    report = json.loads(raw)
    levels = {level: row for level, row in report['levels'].items() if row['model'] is not None}
    if not levels or 'Level6-1' not in levels:
        raise ValueError('Comparable levels including the control are required')
    merged = {}
    for condition in ('greedy', 'sampled'):
        merged[condition] = {}
        for key in ('human', 'policy_probabilities', 'search_visits'):
            values = np.asarray([row['human']['mean'] if key == 'human' else
                                 row['model']['conditions'][condition][key]['mean']
                                 for row in levels.values()])
            if (values.shape != (len(levels), 12) or not np.isfinite(values).all()
                    or (values < 0).any() or not np.allclose(values.sum(axis=1), 1)):
                raise ValueError(f'Invalid distributions: {condition}/{key}')
            merged[condition][key] = values.mean(axis=0).tolist()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13,
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharey=True)
    colors = ['#626A73', '#0072B2', '#D55E00']
    labels = ['NOOP', 'Right', 'Right\n+ jump', 'Right\n+ run', 'Right\n+ jump + run',
              'Jump', 'Left', 'Left\n+ jump', 'Left\n+ run', 'Left\n+ jump + run', 'Down', 'Up']
    x = np.arange(12)
    ymax = max(v for groups in merged.values() for values in groups.values() for v in values) * 100
    limit = max(20, np.ceil((ymax + 2) / 5) * 5)
    for ax, condition, title in zip(axes, ('greedy', 'sampled'),
            ('Greedy rollout states', 'Sampled rollout states (T = 0.25)')):
        for offset, key, color in zip((-.26, 0, .26), merged[condition], colors):
            ax.bar(x + offset, np.asarray(merged[condition][key]) * 100,
                   width=.24, color=color, zorder=3)
        ax.set_title(title, loc='left', fontsize=15, fontweight='bold', pad=10)
        ax.set_xticks(x, labels, fontsize=11)
        ax.tick_params(axis='x', length=0, pad=8)
        ax.set_xlim(-.65, 11.65)
        ax.set_ylim(0, limit)
        ax.set_ylabel('Average share (%)')
        ax.grid(axis='y', alpha=.18, zorder=0)
    fig.suptitle('Human actions, policy head and search', fontsize=25,
                 fontweight='bold', x=.065, ha='left', y=.97)
    fig.text(.065, .912, f'{len(levels)} comparable levels combined · equal weight per level · all outcomes included',
             fontsize=15, color='#535B64')
    fig.legend(handles=[Patch(color=c, label=label) for c, label in zip(colors,
               ('Human actions', 'Policy-head probabilities', 'Search visit probabilities'))],
               loc='upper left', bbox_to_anchor=(.058, .887), ncol=3,
               frameon=False, fontsize=14, handlelength=1.4, columnspacing=2.2)
    fig.subplots_adjust(left=.075, right=.98, top=.785, bottom=.17, hspace=.60)
    fig.text(.075, .058, 'Search visits are measured before action selection; policy and search use the same model states.\n'
             'Humans visit their own states. 5 participants; 30 episodes per level/controller. Level 5-3 lacks model data and is excluded.',
             fontsize=11, color='#535B64', linespacing=1.6)
    destination = args.source.parent / 'probabilities_all_levels_merged'
    fig.savefig(destination.with_suffix('.png'), dpi=240, facecolor='white')
    fig.savefig(destination.with_suffix('.pdf'), facecolor='white')
    plt.close(fig)
    metadata = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'source': str(args.source.resolve()), 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'included_levels': list(levels),
        'excluded_human_only_levels': [level for level in report['levels'] if level not in levels],
        'weighting': 'Equal levels; within level preserve equal human participant/model episode weights.',
        'uncertainty': 'Descriptive means; no uncertainty intervals on the slide.',
        'action_labels': report['action_labels'], 'distributions': merged,
    }
    destination.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved {destination}.png (3840 x 2160), PDF and provenance JSON; {len(levels)} levels.')


if __name__ == '__main__':
    main()
