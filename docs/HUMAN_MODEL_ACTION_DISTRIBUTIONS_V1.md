# Human/model action distributions, 2026-09-11

Figures produced on compute node by job 6457936, from the existing CNeuroMod
converted corpus and completed v1b development trajectories. No confirmation
episodes enter these figures. Plotting and accounting passed the follow-up's
28 targeted tests; both PNGs were visually inspected for labels and clipping.

Artifacts under `outputs/controller_policy/followup-v1-20260911/figures/`:

- `action_distributions.png` and `.pdf`: all 12 discrete action frequencies.
- `button_distributions.png` and `.pdf`: button marginals derived from those
  discrete actions. Simultaneous buttons mean these bars need not sum to 100%.
- `action_distributions.csv`: means, intervals, units and decision counts.
- `action_distributions.json`: full file provenance/hashes, per-participant and
  per-episode frequencies, pooled alternatives, and interpretation limits.

## Counts

Each level includes five human participants (sub-01/02/03/05/06). Each model arm
includes all 30 scheduled development trials. No outcome or participant filtering.

| Level | Human recordings | Human life segments | Human decisions | Greedy decisions | Sampled decisions |
|---|---:|---:|---:|---:|---:|
| 1-1 | 149 | 308 | 125,364 | 43,884 | 18,685 |
| 6-1 control | 150 | 394 | 124,492 | 21,969 | 12,886 |
| 3-2 | 152 | 368 | 116,546 | 23,232 | 17,252 |
| 8-1 | 135 | 375 | 196,634 | 21,099 | 14,934 |

Human frequencies pool decisions within participant and then average the five
participants equally. Model frequencies normalize each episode and then average
the 30 episodes equally. Dots show individual human distributions. Error bars
bootstrap those sampling units, not individual correlated frames. Five humans
provide limited precision for broader population claims.

## A visible Level1-1 difference

Plain-right actions average **75.6%** under greedy, **19.2%** under sampling,
and **23.7%** for humans. Right+jump+run averages **1.1%**, **16.3%**, and **7.9%**
respectively. This supports a descriptive contrast between sustained rightward
input and more varied action combinations. It is consistent with the observed
greedy stalls but does not separate their cause from time spent stuck.

The more varied controller also performs worse on Level6-1 (19/30 versus
28/30 completions). Neither diversity nor closeness of marginal frequencies is a
general measure of competence. On Level8-1 both controllers still complete 0/30.

## Interpretation limits

The human corpus labels nominal four-frame windows using the majority button
vector, then maps that vector to the nearest of the 12 supported actions.
Unsupported combinations can change under this mapping; for example run alone
maps to NOOP under the converter's fewer-buttons tie-break. Therefore the human
NOOP bar does not necessarily mean no physical buttons were pressed, and the
companion button figure is **quantized action-label button use**, not raw 60 Hz
controller usage. Terminal windows can contain fewer than four frames.

Human recordings include respawn segments; model trials use checkpoint reset
settings. Groups visit different positions and situations and differ in time
spent stalled or approaching death. These graphs describe behavior under those
conditions, not actions on matched states or frame-aligned fMRI data. All model
figures use development results and do not independently confirm the selected
sampling controller.
