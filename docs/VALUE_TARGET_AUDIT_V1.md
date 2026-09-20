# Reconstructed value-target audit — 2026-09-20

Question: on actual observations following useful action sequences, is the
frozen value prediction already close to its short-bootstrap target despite
missing the eventual return, or would that target already demand a correction?
This is a diagnostic; no training, checkpoint edits or deployment changes.

## Fixed evidence and controls

Reuse all 56 deduplicated roots from value-commitment-v1-20260918: 46 roots from
23 failed source episodes and ten roots from five successful source episodes.
Level6-1 is mandatory; both levels include intact successful references.
Replay five existing branch arms (continue, logged prefix, release, right+jump,
right+run+jump) and all ten original seeds 7100001–7100010: 2,800 records.
These are existing conditional continuations, not 2,800 independent starts.
Greedy and stall-sampled source groups remain separate. No root noise is added.

The plan retains source hashes and frozen checkpoints. Exact action-sequence
replay verifies every recorded reward, x position, player state, life loss,
completion and terminal flag. Identical trajectories within a root share
inference, retaining each seed's recorded search-Q bootstrap. Predictions use
single-observation FP32 forwards in evaluation mode. Parameters and running
buffers must remain unchanged. All computation runs in Slurm jobs.

## Measurements and interpretation

Checkpoint preflight 6722504 confirms both specialists used discount .999,
10-step value reanalysis, unroll length five and target-network refresh every
200 learner updates. Their saved payloads contain neither the original replay
buffer nor the lagged target network. Existing specialist demonstration files
are post-training evaluations, not original learner data.

Reconstruct targets using the current production target builder and frozen
checkpoint settings. Use the saved online network as an explicitly labelled
proxy for the unavailable lagged target network. This cannot establish the
historical targets, original replay coverage, or original training losses.
Diagnostic continuation policies also differ from noisy training self-play.

At branch positions 0, 1, 2, 4, 8, 16, 32, 64 and 128 (when present), record:

- actual-observation value and observation hash;
- reward window, bootstrap observation index, value and discount factor;
- reconstructed targets with horizons 1, 10, 50, 100 and 200;
- target minus prediction, categorical value loss and support clipping;
- targets using recorded search Q, for comparison with collection-time returns;
- realized discounted remaining return, marking whether it reaches a true terminal.

Primary comparisons use position eight: all prescribed action prefixes have
ended, so the remaining trajectory follows its original continuation controller.
Earlier positions can contain forced future actions; a value difference there
is not automatically a policy-value prediction error. Paths terminating before
eight remain recorded but have no surviving endpoint. Report their counts.
At nonterminal timeouts, recorded return is a finite observed sum, not Monte
Carlo ground truth. Production truncation correctly bootstraps the last saved
observation at its actual distance, even when that distance is zero. Diagnostic
cap is 2,000 decisions; original training collection cap was 5,000.

If the 10-step target stays close to a low prediction while longer targets and
terminal returns rise, short-bootstrap feedback is a plausible mechanism on
these trajectories. If the 10-step target already strongly disagrees with the
prediction, the existing rule could supply a correction if those states were
sampled; fitting, state coverage and policy mismatch remain alternatives.
Neither result alone proves a cause of the original training failure.
Compare these patterns across failure signatures and successful controls;
do not pool away the distinction or infer level-wide reliability.

## Execution

`scripts/submit_value_targets.sh validate OUT` freezes code/config/checkpoints,
runs target-boundary and value-diagnostic tests, then replays one seed for all
five arms at roots 0, 2, 3, 18, 35, 37 and 40. Those cover both levels, successful
controls, sampled-source groups, pit and late-obstacle states. Full arrays run
only after validation succeeds (two concurrent tasks per level, four total).
The final CPU audit requires every root, arm and seed before writing summary.json.

## Storage and validation status

Initial preparation job 6722553 stopped before testing with `Disk quota exceeded`
while freezing source under project outputs. Lustre inode usage was 100%; no
existing result or checkpoint was deleted. The partial v1 directory is retained,
including the completed preflight. Validation 6722600 uses shared home output
`diagnostic_outputs/value-target-audit-v1b-20260920/` instead. Compute remains
on Slurm nodes. This directory is gitignored; original evidence stays on project
storage and is read by hash. Full arrays are not released until validation passes.

Validation **6722600 COMPLETED 0:0 in 2m05s**: all 18 focused tests and all
seven representative replay checks passed (five arms per root). Full Level6-1 array **6722603** (35–55, concurrency two) and
Level1-1 array **6722604** (0–34, concurrency two) depend on validation succeeding.
Final CPU completeness audit **6722606** depends on both arrays. Failed
validation automatically cancels dependent work. Inspect Slurm and logs before
claiming completion; submitted does not mean passed.
