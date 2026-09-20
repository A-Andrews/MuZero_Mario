# Stall-triggered sampling: completed development results

Validation job 6464069 completed 0:0: **21 targeted tests** passed, all nine
smoke episodes finished, and both 6-1 baseline traces exactly reproduced the
original experiment. Full jobs 6464076/6464077/6464078 completed 0:0; all three
summaries have `complete=true`, with **270/270 episodes** and no recorded
infrastructure errors. Read-only compute audit 6468028 verified all 270 trace
hashes and exact greedy/gated equality before the first trigger on all 90 pairs.

Sources: `outputs/controller_policy/stall-sampling-v1-20260911/`, including the
frozen source/checkpoints, `development/<Level>/summary.json`, per-episode JSON
and trace files, and `audit.json`. This is the original **30-seed development
set**, not new independent confirmation. Smoke repetitions are not counted again.

## Completion results

Each cell is completions / 30. All three arms use the same checkpoint, starts,
search seed list, 50 simulations, leaf batch 4, no root noise, and a 2000-decision
limit. Gated sampling uses the fixed 96-decision/16-pixel trigger and up to 32
decisions at temperature .25, with non-control states canceling the burst.

| Level | Greedy | Always sampled | Stall-triggered sampling |
|---|---:|---:|---:|
| 1-1 | 4/30 | 28/30 | **23/30** |
| 6-1 control | 28/30 | 19/30 | **29/30** |
| 1-3 control | 30/30 | 17/30 | **30/30** |

Gated Wilson 95% intervals: 1-1 **59.1–88.2%**, 6-1 **83.3–99.4%**, 1-3
**88.6–100%**. In particular, 30/30 is not proof of perfect reliability.

| Level | Greedy deaths / timeouts | Always-sampled deaths / timeouts | Gated deaths / timeouts |
|---|---:|---:|---:|
| 1-1 | 7 / 19 | 2 / 0 | 7 / 0 |
| 6-1 | 1 / 1 | 11 / 0 | 1 / 0 |
| 1-3 | 0 / 0 | 13 / 0 | 0 / 0 |

## Paired evidence

Against greedy, gated control rescues **19 failures and loses 0 successes** on
1-1, rescues **1 and loses 0** on 6-1, and preserves all 30 successes on 1-3.
Every original timeout on these levels becomes a completion. All original
greedy successes are retained. This supports the proposed separation of stall
handling from normal greedy play on the tested development conditions.

All seven remaining 1-1 gated deaths occur on the same episode indices and at
the same final x positions as greedy, with **zero triggers**: five deaths at
x=898 (indices 4,20,25,26,27) and two at x=1948 (9,19). The remaining 6-1 death
also matches greedy (index 1, x=1411), with zero triggers. The full trace audit
confirms that untriggered gated runs are identical to their greedy counterparts,
not merely similar in final outcome.

This explains an important limitation: the stall controller cannot rescue a
trajectory that dies before it detects a stall. Always-on sampling performs
better on 1-1 in development (28 versus 23 successes); in paired comparisons,
gated rescues one always-sampled failure but loses six always-sampled successes.
Against always-sampled control, gated rescues ten failures and loses none on
6-1, and rescues thirteen and loses none on 1-3. These comparisons do not justify
claiming superiority across arbitrary levels or independently trained models.

## How much behavior changes

| Level | Episodes triggering / 30 | Bursts started | Sampled decisions / all decisions | Pooled sampled fraction |
|---|---:|---:|---:|---:|
| 1-1 | 23 | 45 | 1,255 / 21,790 | 5.76% |
| 6-1 | 10 | 10 | 51 / 15,843 | 0.32% |
| 1-3 | 9 | 9 | 258 / 15,205 | 1.70% |

Mean within-episode sampled fractions are 4.76%, .287%, and 1.58% respectively;
these differ from pooling all decisions because episode lengths differ. A burst
can end before 32 sampled decisions when control state changes or the episode
ends. Thus burst count × 32 is not the number of sampled actions.

The controller does trigger on some successful control-level episodes. Preserved
completion does not mean unchanged behavior everywhere. All gated and greedy
actions, positions, rewards, priors, and search distributions match exactly up
to the first trigger; afterward trajectories may differ. All 6-1 triggers are
at x=2962; 1-3 triggers are at x=2226 or 2419. Level1-1 triggers occur at the
previous staircase locations and several later positions.

Median decisions among successful attempts (greedy / always-sampled / gated):
1-1 **1010 / 636 / 852**; 6-1 **516.5 / 515 / 517**; 1-3
**495.5 / 494 / 495.5**. These medians condition on different successful subsets;
they should not be interpreted as a paired speed benefit for every attempt.

## Conclusion and next test

The fixed controller achieves the intended development behavior: it resolves
the observed timeouts while retaining every greedy success across the two
controls and candidate level. It is a strong candidate for independent
confirmation. The 1-1 early-death signature remains a separate problem.

Keep the rule unchanged and test the reserved **100 new seed pairs** on all
three levels against both baselines. Those ranges (5100001–5100100 and
6100001–6100100) remain unused; no confirmation job was submitted as part of
this results audit. The earlier confirmation data for always-on sampling cannot
validate this newly tested controller. The controller still uses RAM-derived
x/player-state information outside the unchanged image-based network and must
be described as such.
