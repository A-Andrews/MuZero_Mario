# Failure-state action branches

**Completed:** all 4,620 branches and the final trace audit passed after exact
resumption of the quota-interrupted jobs. See `docs/FAILURE_BRANCHES_RESULTS_V1.md`
for the 2026-09-18 results and `docs/FAILURE_BRANCHES_INTERIM.md` for interruption history.

Authorized 2026-09-16 following the completed stall-controller confirmation.
Source: `outputs/controller_policy/stall-sampling-confirmation-v1-20260911/`.
Output: `outputs/controller_policy/failure-branches-v1b-20260916/`.
Template: `docs/failure_branches_v1.json`.

## Questions and fixed selection

Separate 1-1's 11 early deaths, two post-rescue deaths, and one lost greedy
success. Include all nine remaining 6-1 deaths and successful intact approaches
on both levels. No lesion baseline is constructed from an unsuccessful model.

For each of these **23 failed source episodes**, find the first post-action
death-animation state (player_state 11), life decrement, or fall below the
playfield (256 × player_y_screen + player_y_pos ≥480), whichever comes first. Select
the latest saved normal-control observation at or before 32 and 16 decisions
before that onset. This avoids measuring actions during a death animation.
Fail validation rather than silently moving unavailable or duplicate roots.
Vertical RAM is reconstructed by verified full action replay because the old
traces did not retain it. Save y, vertical speed and horizontal speed for review.

For each failed episode choose one successful greedy episode on the same level.
Match each failure root to a normal-control saved state with the nearest x,
requiring eight logged actions remaining. Choose the successful episode with
smallest summed x error across roots, then nearest starting NOOP count, then
lowest episode index. Report x errors and any reused successful episodes. These
are contextual comparisons, not identical states: speed, jump phase and enemies
may differ. No favorable branch outcomes enter selection.

Additionally, for the single lost greedy success, select the last saved normal
state at or before the first gated trigger along its successful greedy trace.
Compare greedy and gated continuations there directly, restoring the monitor's
history from the prefix. The exact old greedy and gated outcomes remain source
evidence; fresh branch seeds are a separate conditional diagnostic.

## Branches and accounting

At each of 92 failure/reference roots use all ten fixed search seeds
7100001–7100010, paired across five arms:

1. Continue the source controller without forced actions.
2. Force the next eight actions from the source log, then continue.
3. Release all buttons for eight decisions, then continue.
4. Right+jump for eight decisions, then continue.
5. Right+run+jump for eight decisions, then continue.

For post-rescue/lost-success failure roots the continuation retains the gated
controller, including its reconstructed internal state. Other ordinary cases
continue greedily. MCTS runs on every decision even during forced prefixes, so
all arms follow the same search-calling convention. Each branch starts a fresh
copy of the same specified RNG stream at its root; later RNG consumption can
diverge with the trajectory. No root noise, 50 simulations, leaf batch 4, frozen
weights. The special trigger comparison has two arms × ten seeds at one root.

Total: **47 cases, 4,620 branches** (4,600 ordinary plus 20 trigger comparisons).
One GPU per array task, at most four tasks concurrently. Record short-term
outcomes at 128 branch decisions and full completion/death/timeout within the
**original 2000-decision episode limit**, subtracting the replayed prefix.
Ten continuation seeds from one state are not ten independent source episodes.
Success-reference reuse and repeated probe points must be visible in analysis.
These are outcome-selected diagnostic states, not an unbiased performance test.

## Replay, predictions and validation

Replay from the original reset seed and actions, checking reset provenance,
every reward/x/state/life count/done flag, and every saved observation through
the original episode. Save RGB clips around selected roots and contact sheets.
Before every branch replay again and require the exact saved root observation.
No partial emulator-state snapshot is used.

At each branch root record the policy prior, value estimate, original action
and visits, fresh search visits, and per-action search return/reward estimates
(unvisited children remain explicitly missing). Observe the original search
tree through a diagnostic wrapper; require exactly equal action, visits, root
value and RNG state against an uninstrumented search from the same seed.
Production model/search/environment code is not edited.

For forced prefixes, also unroll the learned model through the same eight
actions and record predicted rewards and endpoint value. Compare these with
real prefix rewards, the value predicted from the real endpoint observation,
and actual discounted return/completion under continuation. Eight-action
sequence results cannot alone establish which single-action policy ranking is
wrong; learned returns are not completion probabilities and continuation
policies can differ from the policy implicit in the learned value target.

Compute validation runs selection/replay/monitor tests and short smoke branches
for the first case of every level/signature combination, including successful
references and the trigger comparison. Smoke uses the first fixed diagnostic
seed in separate output and is not added to full denominators. Full array tasks
depend on successful validation; a CPU aggregation and trace-hash audit depends
on every array task succeeding. Missing results are never replaced by new seeds.

This consumes the confirmation data for further diagnosis. Any subsequently
tuned controller needs new evaluation seeds. No policy/BatchNorm training
change is included in this experiment.

## Validation correction before full submission

Initial validation 6602385 passed 16 tests but its source contact sheets revealed
that the death-state/life-loss signal was too late for pit falls: Mario was
already off-screen at the proposed lost-success roots. No full array was
submitted from that plan. The revised v1b rule includes the below-playfield
signal for every failure, preserving the failed episode set, offsets, arms and
seeds. Both frozen setups remain available; the v1 smoke outcomes are not added
to the revised experiment. This is a correction to diagnostic-state selection,
not evidence about controller performance.

## Submitted jobs and validated scope

- Corrected validation **6602469** completed 0:0: **17 tests**, seven complete
  source replays with observation/trajectory checks, and **32 short smoke
  branches** passed. Search instrumentation preserved results and RNG state.
  Preparation also verified vertical-RAM replay for all 23 failed episodes.
- Full array **6602494**, tasks 0–46, maximum four concurrent GPUs, depends on
  successful corrected validation.
- CPU aggregation/trace audit **6602502** depends on every full array task.

Corrected contact sheets for cases 14 and 28 show Mario on-screen before the
fall at the selected approach windows. The final plan contains five distinct
successful source episodes, reused for matching: 1-1 episodes 6, 47, 52 and
6-1 episodes 21, 93. Repeated reference roots/episodes are not independent
evidence. The array evaluates the declared comparisons; any reporting must
retain these shared-source identities. Full results were not available when
the array was submitted.
