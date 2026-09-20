# Controller and policy diagnostics — development protocol v1

**Completed:** all jobs succeeded and all 600 development trials were collected.
See [CONTROLLER_POLICY_RESULTS_V1.md](CONTROLLER_POLICY_RESULTS_V1.md) for the
2026-09-11 results audit; the submission notes below retain their original status.

This experiment follows the user's 2026-09-07 request to investigate policy-head
learning while gathering evidence for a stochastic Mario controller. It changes
evaluation and adds probes; production training, checkpoints, and MCTS are unchanged.

## Questions and interpretation

1. Does action sampling, root noise, or sequential search change completion under
   matched starts? Compare five frozen controllers on Level1-1, Level3-2,
   Level8-1, and the successful Level6-1 control.
2. Does the head fit its current search targets? Measure cross-entropy, target
   entropy, and their difference KL(target || prior) on the same observations.
   A flat target with a matching flat prediction has zero KL. High cross-entropy
   alone is not evidence of underfitting.
3. Is there state information? Compare correct observation/target pairs against
   within-episode shuffled pairs and a constant mean policy. These controls are
   descriptive: search itself uses the prior, so agreement is not proof that
   actions are good.
4. Is flatness mechanically induced? Re-search states with leaf_batch 1 and 4 at
   50 simulations. Repeat noisy search five times on eight states to separate
   agreement with the mean target from variation among random targets.
5. Does inference depend on BatchNorm statistics? Compare checkpoint statistics
   with batch statistics on isolated model copies, varying batch composition.
   This does not reproduce the learner's mix of real and imagined states and
   does not propose using arbitrary batch statistics as a deployed controller.

Original training replay was RAM-only and is unavailable from these checkpoints.
The new banks contain fresh frozen-checkpoint evaluation targets. Their fit
cannot establish historical optimization quality. The final logged policy loss
already beats a uniform predictor: root CE + mean recurrent CE is about 3.17
(1-1) and 3.12 (6-1), against 2 ln(12) = 4.97 for uniform predictions. Those
final-run metrics are not measurements of best.pt at its stall.

## Predeclared controller comparison

The tracked `controller_policy_diagnostic_v1.json` fixes all development trials.
Each condition receives 30 attempts per level, at 50 simulations and a
2000-agent-step limit, with each checkpoint's stochastic-start configuration.

| Condition | Leaf batch | Root epsilon | Temperature |
|---|---:|---:|---:|
| greedy | 4 | 0 | 0 |
| sequential_greedy | 1 | 0 | 0 |
| root_noise | 4 | .25 | 0 |
| sampled | 4 | 0 | .25 |
| noise_and_sampling | 4 | .25 | .25 |

Dirichlet alpha is .25. Every episode uses a new environment instance. Trial i
uses environment seed 1100001+i and separate search seed 2100001+i. Controller
conditions use the same trial pairs. Exact starting fingerprints are checked;
actual requested NOOP counts are recorded. Different seed numbers do not imply
different initial states: the reset distribution has only 31 NOOP offsets.
MCTS still randomly breaks exact UCB ties, even when final action choice is greedy.

All outcomes count. Infrastructure errors stop the job and remain recorded;
resumption retries the same scheduled trial, never substitutes another seed.
Greedy cells retain the full trial count. Results include completion counts,
Wilson intervals, deaths, timeouts, and pending counts. Partial results are
explicitly interim. Episodes are interleaved across controller conditions.

The manifest reserves 100 separate confirmation seed pairs (3100001+i and
4100001+i). No confirmation trials run in this experiment. Select the controller
on development results, freeze that choice, and use the reserved trials once for
confirmation. These trials assess the selected checkpoints on this start
distribution, not reproducibility over independently trained models or unseen levels.

## Artifacts and execution

`prepare_controller_diagnostic.py` runs on a compute node and copies checkpoints
and source into a new experiment directory under `/projects`. The resolved
manifest records SHA256 hashes, the source snapshot, and original checkpoint paths.
Existing experiment directories cannot be silently overwritten. Jobs execute the
frozen source. The evaluator verifies checkpoint and resume identities.

Each episode writes a result JSON, a compressed full decision trace, and a
compressed observation bank (every 20th state plus the last 200 states). Banks
include failed attempts. Policy probes sample multiple episodes, progress
regions, and episode tails, reporting episode counts separately from correlated
state counts. BatchNorm regroupings do not count as independent episodes.

Submit `scripts/submit_controller_policy_diagnostic.sh prepare-smoke OUT` first,
with a one-hour limit. It prepares the snapshot, runs targeted tests, then one
episode per condition on Level1-1 and Level6-1 plus both policy probes. Those
smoke trials are development data and must not be added again to the 30-trial
denominator. After validation, submit `full OUT LEVEL` for each declared level.
All model inference and tests run on Slurm compute nodes.

## Corrections to earlier T11 interpretation

- Level3-2 and Level6-3 each completed 4/5 greedy benchmark trials. A zero
  failure-position spread from a single failed trial does not establish clustering.
- Level8-1's historical best rate is .01 and its noisy collection was 0/60;
  successful noisy competence has not been established there.
- Flat visits also occur on successful Level6-1. They are not sufficient to
  explain failure, although a shared search issue can matter differently at
  different obstacles.
- Policy lesions establish dependence on the head, not poor learning in the
  intact head. The transition-model lesions also caused large deficits.

## Submitted execution

The active frozen experiment is
`outputs/controller_policy/v1b-20260907/` (the `/projects` output symlink).
Validation job **6386365** passed all **40 targeted tests**. Its first Level1-1
greedy smoke trial timed out at 2000 steps, x=2370, reproducing the existing
failure region. Other smoke trials and probes were still running when this
status was recorded; n=1 is not a controller comparison.

Full development jobs are conditional on validation succeeding:

| Level | Job |
|---|---:|
| Level1-1 | 6386532 |
| Level6-1 control | 6386533 |
| Level3-2 | 6386534 |
| Level8-1 | 6386535 |

They use `afterok:6386365` with `kill-on-invalid-dep=yes`; failed validation
cancels the dependent jobs rather than allowing unvalidated trials. Each full
job has one GPU and a 12-hour ceiling. Results and logs are in the experiment
directory, with per-level development `summary.json` indicating completeness.
Read actual job/result status before interpreting these submitted jobs as done.

Earlier preparation job **6386263** passed 39 tests but failed before gameplay
on `/projects` versus `/lus/...` source-path normalization. Its original
`v1-20260907/` snapshot remains for provenance. The corrected script resolves
both paths, with a regression test; development trial seeds did not change.
