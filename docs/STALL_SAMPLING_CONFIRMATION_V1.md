# Stall-triggered sampling: independent confirmation

**Completed:** all 900 episodes and the trace audit passed. See
`docs/STALL_SAMPLING_CONFIRMATION_RESULTS_V1.md` for the 2026-09-16 results review.

Authorized 2026-09-11 after the completed 30-trial development investigation.
The candidate is unchanged: normal player state 8, x span at most 16 pixels
over 96 elapsed decisions, then up to 32 decisions sampled at temperature .25.
Afterward return to greedy and clear the window; non-control states cancel the
burst. RAM x/player-state are additional controller inputs outside the visual
network. No training, weight, normalization, or search changes are included.

## Fixed evaluation plan

- Levels: 1-1, successful 6-1 control, successful 1-3 control.
- Arms: greedy, always sampled at T=.25, stall-triggered sampling.
- Exactly 100 paired attempts per arm per level: **900 new episodes**.
- Environment seeds 5100001–5100100; search seeds 6100001–6100100, copied from
  the previously frozen development manifest's reserved list.
- All three arms use identical starting seeds and frozen checkpoint identities,
  50 MCTS simulations, leaf batch 4, no root noise, and a 2000-decision limit.
- All attempts, including deaths and timeouts, count. No seed substitutions,
  early stopping, parameter tuning, or selecting favorable subsets.
- Earlier 30-trial development and earlier 100-trial two-arm confirmation remain
  separate. The latter's seeds 3100001/4100001 have already been consumed and
  are excluded here. Both baselines run afresh on the new paired seeds.

Template: `docs/stall_sampling_confirmation_v1.json`.
Output: `outputs/controller_policy/stall-sampling-confirmation-v1-20260911/`.

## Validation and execution

The compute validation job freezes source/checkpoints, runs protocol and
controller tests, verifies that the seed reserve, controller settings, model
source, environment source, and search source match development, then runs nine
smoke episodes on **old development seeds only**. All three controller traces
on all three levels must reproduce their old episode-zero traces exactly.
Confirmation seeds cannot be used for smoke tests or partial trial overrides.

Three full compute jobs depend on successful validation, one per level. A final
compute audit depends on all three succeeding; it checks 900 trace hashes and
greedy/gated equality before the first intervention for all 300 matched pairs.

## Planned interpretation

Report completion counts / 100 and Wilson 95% intervals separately for each
level and controller, plus deaths and timeouts. Compare paired rescues and lost
successes against both baselines. Report how many greedy timeouts become
completions, remaining deaths before intervention, triggered episodes, and
sampled decision fractions. The key questions are whether 1-1 stalls are still
rescued and whether the successful controls retain their greedy successes.
Finite samples cannot establish zero risk even if no greedy successes are lost.
This confirms a fixed-checkpoint controller, not a repaired learned policy or
general performance across all levels or independently trained models.

Submitted compute jobs:

| Job | Purpose | Dependency |
|---|---|---|
| 6469923 | Freeze, tests, old-seed smoke and replay validation | None |
| 6469925 | 1-1, 300 confirmation episodes | Validation succeeds |
| 6469926 | 6-1, 300 confirmation episodes | Validation succeeds |
| 6469927 | 1-3, 300 confirmation episodes | Validation succeeds |
| 6469932 | CPU trace and paired-outcome audit | All three full jobs succeed |

All five jobs completed 0:0. Validation passed 22 tests and nine exact old-seed
replays. The final audit verified all 900 trace hashes and 300 paired prefixes.
