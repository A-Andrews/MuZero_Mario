# Speaker notes — approximately 18 minutes

## 1. MuZero–Mario: what works, what fails, and how to resume

0:00–1:00. State the purpose: preserve the project and agree on the next experiment. The human reference is the recorded CNeuroMod cohort, not expert speedrunners. This talk reports existing experiments, not newly trained models.

## 2. What would human-level performance mean?

1:00–2:30. Five human participants, repeated practice and scanner conditions. Human/model starts and time caps differ. Human successful duration bands include respawn lives. Avoid one human-normalized headline ratio. Explain uncertainty conditions on frozen checkpoints.

## 3. Inside MuZero: observe, imagine, search, act

2:30–4:00. Explain representation, dynamics, policy prior, value, and MCTS. The model predicts task-relevant hidden states, not screenshots. One executed action lasts four emulator frames; the system replans afterward. Values are expected shaped rewards, not winning probabilities. Benchmark search uses 50 simulations, leaf batch four.

## 4. What the project has built

4:00–5:00. Coverage of representations is not proof of competent gameplay. Flag imitation-trained models in downstream brain comparisons. T7 terminal noisy self-play metrics are not matched evaluation scores; T8 has no established completed student result.

## 5. Where we stand against recorded humans

5:00–7:00. This is the all-level development comparison, 30 attempts/controller/level. The lower panel divides successful model median duration by the human successful median; human rate diamonds pool repetitions. Point to strong levels and unresolved failures; 5-3 is missing, not zero. Use the separate PDF to inspect fine detail. Do not infer general human-level play from isolated point estimates.

## 6. Removing stalls is not enough

7:00–8:00. On the active 21-model-level set, 203 versus 202 completions out of 630. Always-on sampling eliminates 90 timeouts but adds 91 deaths. The older 22-model-level total included 2-2; it is not this cohort.

## 7. A practical improvement that replicated

8:00–9:30. Independent 100-pair confirmation, 900 episodes. Stall-triggered sampling uses extra RAM x/player-state input, with unchanged weights. It rescues 76/78 greedy timeouts on 1-1 but loses one success. 86 versus 83 is not established superiority. No all-level gated result yet.

## 8. Watch the same recorded start under two controllers

9:30–10:30. Play the baseline and gated clips. Explain that clips illustrate the mechanism; aggregate tables estimate rates. Additional early-death, successful-control and lost-success clips are in clips/. If video is unavailable, show slide 7 and describe the exact endpoints.

## 9. Action timing can help—and harm

10:30–12:00. Holding the initial action rescues the late 6-1 obstacle roots, but harms successful references. The imagined/observed value discrepancy is a clue, not ground truth and not automatically the discrepancy encountered by search. Ten seeds at a selected root are conditional repeats.

## 10. Values on actual searched paths: a selective causal effect

12:00–13:30. Eight-decision substitution rescues 6-1 pit roots, not obstacle roots. Some 1-1 successes are lost. No terminal paths were encountered, so terminal-value replacement cannot explain these effects. Changed backups also alter future search allocation; the intervention does not isolate dynamics versus representation/value-head effects.

## 11. What the completed value-target audit adds

13:30–15:00. Audit complete: 56 roots and 2800 reused records. These are illustrative endpoints, not a population test. At root 40 a 10-step proxy target is already 131 against prediction 18; at another pit root it remains low. Historical target network and training replay are unavailable, so this cannot prove the original training cause.

## 12. The next work should answer three decisions

15:00–17:00. The experiment cards in NEXT_EXPERIMENTS.md specify inputs, boundaries, outputs and decision rules. Finish the protocol for broader confirmation before consuming fresh seeds. Do not launch all experiments at once. Prefer targeted pilots and intact controls.

## 13. How someone can pick this up

17:00–18:00. Present verification results and distinguish clean plotting setup from tested existing cluster inference environment. The access checklist is user-specific; a test under the current account cannot prove a colleague has access. Agree ownership, first experiment and a review date.
