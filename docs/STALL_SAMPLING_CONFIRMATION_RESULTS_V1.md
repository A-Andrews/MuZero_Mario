# Stall-triggered sampling: completed independent confirmation

Audited 2026-09-16. Validation 6469923, evaluations 6469925/6469926/6469927,
and final trace audit 6469932 all completed with exit code 0:0. All **900/900**
scheduled episodes finished, with no recorded infrastructure errors. Validation
passed 22 tests and nine exact old-seed controller replays. The final audit
verified 900 trace hashes and exact greedy/gated behavior before intervention
on all 300 matched pairs.

Source: `outputs/controller_policy/stall-sampling-confirmation-v1-20260911/`,
specifically `confirmation/<Level>/summary.json`, individual episode records,
`control_validation.json`, and `audit.json`. Protocol:
`docs/STALL_SAMPLING_CONFIRMATION_V1.md`. These are the 100 previously reserved
new seed pairs, not the 30 development pairs or the earlier two-arm confirmation.
The complete fixed list was evaluated without tuning the controller.

## Completion and failure counts

Each count is out of 100. The candidate uses the unchanged model and greedy
MCTS except for up to 32 sampled decisions at T=.25 after the fixed 96-decision,
16-pixel stall trigger. All arms disable root noise.

| Level | Greedy | Always sampled | Stall-triggered | Stall-triggered Wilson 95% interval |
|---|---:|---:|---:|---:|
| 1-1 | 11 | 83 | 86 | 77.9–91.5% |
| 6-1 control | 84 | 80 | 91 | 83.8–95.2% |
| 1-3 control | 100 | 55 | 100 | 96.3–100% |

| Level | Greedy deaths / timeouts | Always-sampled deaths / timeouts | Stall-triggered deaths / timeouts |
|---|---:|---:|---:|
| 1-1 | 11 / 78 | 17 / 0 | 14 / 0 |
| 6-1 | 9 / 7 | 20 / 0 | 9 / 0 |
| 1-3 | 0 / 0 | 45 / 0 | 0 / 0 |

## Paired outcomes and remaining failures

On 1-1, stall-triggered sampling rescued **76 of 78 greedy timeouts**, retained
10 of 11 greedy successes, and retained all 11 pre-intervention deaths. The two
other timeouts became later deaths. Thus the development observation of zero
lost greedy successes does not hold universally: **one success was lost**.

That lost success is episode index 45: greedy completes in 1269 decisions;
gated triggers at decisions 394 and 547 (x=2226 and 2354), then dies at x=2582
after 657 decisions. The two timeout-to-death cases are indices 11 and 59:
both escape the old x=2354 timeout location, then die at x=2755 and 2585.
The other 11 gated deaths have zero triggers and identical greedy trajectories:
nine at x=898 and two at x=1948. These remain a separate failure signature.

On 6-1, all seven greedy timeouts become completions and all 84 greedy successes
are retained. The nine deaths occur before any trigger and exactly match greedy:
six at x=1411 and three at x=2807. On 1-3, all 100 greedy successes are retained.
These controls support preserving successful greedy play while intervening at
stalls, within the tested checkpoints and start distribution.

Against always-sampled control, paired rescues / losses are 14 / 11 on 1-1,
19 / 8 on 6-1, and 45 / 0 on 1-3. The 86 versus 83 result alone does not
establish superiority on 1-1; there are substantial disagreements in which
trials succeed. Always-on sampling's cost on 1-3 is clear in this sample.

## Intervention frequency

| Level | Triggered episodes / 100 | Bursts | Sampled decisions / total | Pooled sampled fraction |
|---|---:|---:|---:|---:|
| 1-1 | 86 | 163 | 4539 / 77684 | 5.84% |
| 6-1 | 27 | 27 | 124 / 51431 | 0.24% |
| 1-3 | 29 | 29 | 868 / 51265 | 1.69% |

Mean within-episode sampled fractions are 5.24%, 0.21%, and 1.55% respectively.
Bursts can stop before 32 decisions when control state changes or an episode ends.
Preserved completions do not mean behavior remains unchanged after intervention.

## Interpretation and next diagnostic

Independent confirmation supports a practical stall-controller benefit. It does
not show that the policy head has been repaired, establish normalization as a
cause, or resolve early deaths. Model weights are unchanged. The controller
uses RAM-derived x/player state outside the image-based network and should be
reported with that additional input. Reliability is estimated for these frozen
checkpoints, not all Mario levels or independent training runs.

The next useful diagnostic is to examine the early-death trajectories separately
from the single lost success and two post-rescue deaths, using matched successful
trajectories and Level6-1 as control. Do not tune the current candidate on this
confirmation set and continue describing the same set as untouched confirmation.
No additional experiment was submitted during this results review.
