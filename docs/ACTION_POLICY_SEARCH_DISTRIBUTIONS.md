# Human actions, policy head and search distributions

Requested 2026-09-18. Compute job **6671364** reads the existing converted
CNeuroMod corpus and frozen evaluation traces; it does not run new model
inference or gameplay. Script: `scripts/plot_policy_search_distributions.py`.
Outputs: `images/action_policy_search/`.

**Completed:** job 6671364 finished 0:0 in 58 seconds; all three focused tests
passed and all source/trace checks passed. All 22 human levels were collected;
21 have 30 model episodes per condition (1,260 model episodes total). The
1-1 detail page and all-level greedy overview were visually checked.

## Deliverables

- `probabilities_all_levels_merged.png`: 3840 × 2160 slide combining the 21
  levels with both human and model data, with separate greedy/sampled panels.
  Levels have equal weight; within each level the participant/episode weighting
  below is preserved. Human-only 5-3 is excluded from all three groups in this
  comparison. Descriptive means without uncertainty bars. PDF and provenance
  JSON companions included. Render with `scripts/plot_merged_policy_search.py`
  on a compute node using the existing `distributions.json` (job 6671714).
- `probabilities_by_level.pdf`: one page per human-covered level, with separate
  greedy and sampled rollout panels. The three groups are observed human
  action labels, averaged policy-head probabilities, and averaged raw MCTS
  visit probabilities. Per-level PNGs are also provided.
- `probabilities_overview.pdf`: two heatmap pages, greedy and sampled, showing
  the same comparison across every level; corresponding overview PNGs included.
- `action_choices_by_level.pdf`: companion pages comparing human actions,
  hypothetical policy-head argmax choices on logged model states, and the
  actions actually executed after search.
- `distributions.csv` and `distributions.json`: means, participant/episode
  bootstrap intervals, unit-level distributions, counts and source provenance.

The corpus provides 22 human-covered levels, each with five participants.
Model data cover 21 levels using all 30 greedy and 30 sampled development
episodes from `controller_policy/v1b-20260907` (four levels) and
`controller_policy/coverage-v1-20260911` (remaining levels). The earlier 2-2
model results are excluded because no human gameplay exists. 7-2 is absent
from the corpus. **5-3 has human data and is shown with model data unavailable**;
missing model values are not rendered as zero.

## Meaning and weighting

Policy and search distributions are evaluated on exactly the same saved model
observations within each controller condition. This makes their change through
search directly interpretable descriptively. Greedy and sampled trajectories
visit different states and are displayed separately. Humans visit their own
states, with different starts and respawn histories; these plots are not
matched-human-state accuracy tests or evidence of fMRI alignment.

For each participant, pool all their human decisions, including failures and
respawn segments, then normalize. Average participants equally. For models,
average probabilities or action counts within each complete episode, then
average all scheduled episodes equally. Thus a long stall does not receive
more weight than an entire short episode. All outcomes remain included.
Bootstrap intervals resample participants or episodes (4,000 draws), never
individual frames. Individual human distributions appear as dots.

Raw search visits are the pre-temperature policy training target; they are
not executed-action frequencies. The model chooses greedily at T=0 or samples
at T=.25 after search, with no root noise, 50 simulations and leaf batch 4.
The companion argmax plot is hypothetical on the recorded observations; it
is not a rollout performed by the policy head alone.

Human labels are the existing converter's majority-button choices over nominal
four-frame windows, mapped to the nearest supported model action. They are
not original per-frame button frequencies. All 12 actions, including
right+jump+run, remain visible.

Checks reject incomplete/wrong-split evaluations, altered trace hashes,
checkpoint/seed/record identity mismatches, non-normalized probabilities, and
misaligned prior/visit/action arrays. Greedy executed actions must equal
search-visit argmax. Three focused tests cover probability-versus-argmax
semantics, equal episode weighting and invalid probability inputs.
