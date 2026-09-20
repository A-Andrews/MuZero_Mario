# Policy investigation follow-up, 2026-09-11

**Completed:** both reserved-seed confirmation jobs and their figure succeeded.
See [CONTROLLER_INVESTIGATION_RESULTS_20260911.md](CONTROLLER_INVESTIGATION_RESULTS_20260911.md)
for the final 100-trial results; running-status notes below describe submission time.

Authorized after the development results review. Production training, MCTS and
checkpoint weights are unchanged. All inference, gradient computation, tests and
corpus analysis run on Slurm compute nodes. The user selected the existing human
corpus, all participants, and all four diagnostic levels for the figures.

## Reserved-seed controller confirmation

Freeze greedy (leaf batch 4, epsilon 0, temperature 0) versus sampling-only
(leaf batch 4, epsilon 0, temperature .25), each with 50 simulations and a
2000-decision cap. Use exactly the original 100 reserved environment/search seed
pairs on Level1-1 and Level6-1: 400 total episodes. Both use the same best.pt
copies as development, with expected SHA256 checks before preparation.
`controller_policy_confirmation_v1.json` preserves the development seed list,
reserved seed list, checkpoint identities and controller settings. No seed
replacement, outcome filtering, tuning on confirmation, or early stopping.
Level1-1 is the selected candidate level; Level6-1 quantifies the known trade-off.
This does not validate a universal controller or independent training runs.

Smoke runs use development seeds in a separate directory. The evaluator rejects
confirmation smoke runs and trial/condition overrides. Confirmation reports carry
their own split and are never pooled with development. Report paired rescue/loss
counts, completion Wilson intervals, deaths and timeouts once all trials finish.

## Gradient balance probe

Use existing development greedy and sampled trajectories separately for Level1-1
and Level6-1. Select up to 16 root observations per episode across elapsed time,
using all 30 episodes per arm (at most 480 roots). Read the next K actual actions
and K+1 search policy targets from the complete matching trace. This avoids
mistaking sparse bank rows for consecutive time steps. Exclude roots without K
future decisions; record exclusions and selected steps. Episodes of every outcome
are eligible. Each batch contains at most one root per episode.

Reproduce the learner's policy objective: root cross-entropy plus the mean of K
recurrent cross-entropies, including its recurrent hidden gradient hook of .5.
Measure both terms' gradient norms, cosine, and recurrent projection onto the
root gradient separately for the policy head, shared prediction trunk,
representation, and dynamics trunk. The root loss has no path to dynamics, so its
zero gradient there is expected, not evidence of imbalance.

Compare checkpoint BatchNorm statistics with current-batch statistics. Both keep
BatchNorm and latent min-max normalization. Also repeat the batch-stat probe with
hook factor 1 to check what the hook actually scales. There is no optimizer step;
assert all parameters and running buffers remain identical. Batch-stat mode uses
the actual real-root and recurrent-imagined forwards, unlike the preceding
real-observation-only BN probe.

This measures policy gradients on fresh evaluation data in fp32 with equal
sample weights. It excludes value, reward, consistency losses, prioritized replay,
AMP, gradient clipping and Adam state. It cannot reconstruct historical training
or establish a behavioral cause. Different batches reuse the same episodes;
report batch ranges and episode counts, not independent-state significance.
Large recurrent gradients could reflect larger prediction errors; negative cosine
indicates opposing local policy-loss directions, not necessarily harmful training.

## Human/model action figures

Use every available converted human segment on Level1-1, Level6-1, Level3-2 and
Level8-1, including failures and truncations, and all 30 development trials for
each model controller (greedy and sampling-only). Human labels use the existing
converter: majority button states over nominal four-frame windows, mapped to the
nearest supported discrete action. This is not raw 60 Hz button timing.

Pool decisions within each participant, then average participants equally. For
models, normalize within each episode, then average episodes equally. This keeps
one prolific participant or long stalled episode from dominating the comparison.
Also export pooled-decision frequencies for transparency. Human recordings can
include multiple respawn segments and start differently from model episodes.

Produce a 12-action figure and a companion button-use figure derived from those
same quantized actions, in PNG/PDF, plus CSV and JSON with file hashes, counts and
sampling-unit frequencies. Human dots show individual participants; intervals
resample participants for humans and episodes for models. With only five humans,
intervals give limited population precision. Buttons are not mutually exclusive.

These are marginal behavior distributions on each group's own visited states.
Differences may reflect stalls, movement speed, state coverage or action choice.
They are not matched-state policy accuracy, human-level performance evidence,
independent confirmation data, or a frame-aligned fMRI analysis.

## Execution

`scripts/submit_policy_followup.sh validate OUT OLD` freezes source/checkpoints,
runs targeted tests, creates the human action figures, and smoke-tests both probes
on development data. Submit `confirmation OUT OLD LEVEL` and `gradients OUT OLD
LEVEL` with an afterok dependency on validation and kill-on-invalid-dep enabled.
Full gradient jobs use a two-hour limit; confirmation jobs have a twelve-hour cap.
Output and source snapshots live under `/projects`, outside the home quota.

Submitted output: `outputs/controller_policy/followup-v1-20260911/`, using
`outputs/controller_policy/v1b-20260907/` as the immutable development source.

| Task | Job |
|---|---:|
| Validation, figures, smoke probes | 6457936 |
| Level1-1 confirmation | 6457937 |
| Level6-1 confirmation | 6457938 |
| Level1-1 full gradient probe | 6457939 |
| Level6-1 full gradient probe | 6457940 |

Validation completed successfully (0:0, 5m48s): 28 targeted tests, both PNG/PDF
figures and CSV/JSON, both gradient smoke probes, and both controller smoke arms
on both levels. Smoke outcomes/lengths/positions/starts matched their original
development trial 0 counterparts. The full gradient jobs also completed (0:0),
each covering 30 episodes and 480 roots per controller. See
`POLICY_GRADIENT_RESULTS_V1.md` and `HUMAN_MODEL_ACTION_DISTRIBUTIONS_V1.md` for
the completed audits. The two confirmation jobs are running; inspect actual
Slurm status and result `complete` fields before claiming their completion.
