# Root/recurrent policy gradient results, 2026-09-11

Full compute jobs 6457939 (Level1-1) and 6457940 (Level6-1) completed with exit
code 0. Both reports have `complete=true` and verified unchanged parameters and
running buffers. Source and checkpoints are frozen under
`outputs/controller_policy/followup-v1-20260911/`; reports are
`gradients/Level1-1.json` and `gradients/Level6-1.json`.

Each level/controller condition used all **30 development episodes**, with
**480 selected root states** in **16 batches of 30**, one root per episode per
batch. Greedy and sampling-only are separate. The same batches are reused for
normalization/hook comparisons; they are not additional independent samples.

## Policy head: no evidence of recurrent domination specific to Level1-1

The learner weights root CE and the mean of five recurrent CEs equally. The
table measures their actual gradients using batch statistics (training forward
normalization) and the existing .5 recurrent hidden gradient hook. Ratios and
cosines are medians across batches; ranges are descriptive, not intervals.

| Level | Trajectory controller | Recurrent/root norm ratio | Ratio range | Root/recurrent cosine | Negative-cosine batches |
|---|---|---:|---:|---:|---:|
| 1-1 | Greedy | .812 | .257–1.460 | .387 | 0/16 |
| 1-1 | Sampled | .924 | .432–2.117 | .539 | 0/16 |
| 6-1 control | Greedy | 1.219 | .250–7.059 | .300 | 1/16 |
| 6-1 control | Sampled | 1.142 | .490–4.297 | .316 | 0/16 |

On the failing Level1-1 greedy distribution, the averaged recurrent policy
gradient is typically smaller than the real-root gradient and always positively
aligned in these batches. Larger ratios and occasional opposition also occur on
the successful Level6-1 greedy baseline. This weakens the specific hypothesis
that imagined-step policy gradients overwhelm the Level1-1 policy head and cause
its stalls. It does not prove that the full training process is well balanced.

The shared prediction trunk shows median ratios .891/.866 (1-1 greedy/sampled)
and 1.043/1.071 (6-1 greedy/sampled). Negative cosines occur in 2/0/1/1 of the
16 batches respectively. Some local conflict exists; it is not unique to the
failing controller/level.

## The .5 hook mainly protects upstream representations

Removing only the recurrent hidden gradient scaling leaves every measured
policy-head and shared-prediction recurrent gradient norm unchanged (maximum
absolute difference 0). This matches the computation graph: those prediction
parameters are downstream of the scaled hidden state.

Representation gradients change considerably. Median recurrent/root ratios:

| Level | Controller | Existing .5 hook | Hook removed (factor 1) |
|---|---|---:|---:|
| 1-1 | Greedy | .184 | .967 |
| 1-1 | Sampled | .167 | .899 |
| 6-1 control | Greedy | .248 | 1.318 |
| 6-1 control | Sampled | .290 | 1.459 |

The effect can exceed a factor of two because gradients from later losses cross
multiple recurrent hidden states. Dynamics-trunk gradients also grow when the
hook is removed. The real-root policy loss has no path into dynamics, so its
dynamics gradient is correctly zero and a recurrent/root ratio there is undefined.
This supports keeping the hook while investigating other failure mechanisms.

## BatchNorm changes gradients but is not isolated as a failure cause

With checkpoint running statistics, policy-head root/recurrent cosine medians
are .961/.939 for 1-1 greedy/sampled and .787/.684 for 6-1. Current-batch
statistics reduce alignment on both levels (table above). This is sensitivity to
normalization context shared by the successful control, not a Level1-1-specific
mechanism. Both modes retain BatchNorm and per-sample hidden min-max normalization;
no normalization layer was removed and no BN-modified controller was evaluated.

## Limits and next interpretation

These are policy-only gradients on **fresh frozen-checkpoint evaluation traces**,
not original training replay. Root observations use real frames; recurrent
predictions use the actual next five actions and targets from their matching
complete trace. Only roots with a full nonterminal unroll are included.

The probe uses fp32 and equal sample weights. It excludes value/reward/consistency
objectives, prioritized sampling, optimizer state, clipping, weight decay and
historical learning dynamics. Neither a large gradient nor a negative cosine
alone establishes harmful training. The successful Level6-1 baseline remains
necessary when testing broader multi-loss interference or normalization changes.

Reserved-seed controller confirmation jobs 6457937/6457938 are running separately;
these gradient results do not consume or analyze their data. Current evidence
favors investigating local action selection and stall transitions before changing
policy-loss weights or BatchNorm on the basis of recurrent-count arguments.
