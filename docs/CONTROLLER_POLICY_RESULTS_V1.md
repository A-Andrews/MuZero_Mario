# Controller and policy results, v1

Read on 2026-09-11 from the frozen experiment
`outputs/controller_policy/v1b-20260907/`. Validation job 6386365 and development
jobs 6386532–6386535 all completed with exit code 0. Validation passed 40 tests.
All four development summaries have `complete=true`: 600 scheduled rollouts,
30 per condition per level, with no recorded infrastructure failures.

These are development results from four fixed specialist checkpoints. The 100
reserved confirmation trials have not run. Do not select settings on these
results and describe the same results as an independent confirmation.

## Completion results

Every cell below is completions / 30 attempts. Noise means root Dirichlet
epsilon .25 and alpha .25; sampling means visit-count temperature .25. All
conditions use 50 simulations. Leaf batch is 4 except sequential greedy (1).

| Level | Greedy | Sequential greedy | Root noise only | Sampling only | Noise + sampling |
|---|---:|---:|---:|---:|---:|
| Level1-1 | 4/30 | 13/30 | 23/30 | 28/30 | 23/30 |
| Level6-1 control | 28/30 | 12/30 | 26/30 | 19/30 | 19/30 |
| Level3-2 | 25/30 | 23/30 | 21/30 | 29/30 | 18/30 |
| Level8-1 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |

Level1-1 greedy failures include 19 timeouts and 7 deaths. Sampling alone
produces zero timeouts and 2 deaths; it rescues 24 of the same trial indices
where greedy failed and loses none where greedy succeeded. Wilson 95% intervals
are 5.3–29.7% for greedy and 78.7–98.2% for sampling. These intervals describe
the tested start/search distribution, not arbitrary situations or training runs.
Root noise is not necessary for the observed rescue. The proximal evidence
supports an action-selection-dependent stall, but does not isolate which local
alternative action or sequence releases the trap.

Level6-1 rules out a universal controller change. Sampling loses 10 paired
greedy successes and rescues 1 greedy failure. Sequential search loses 18 paired
greedy successes and rescues 2 failures; it creates 17 timeouts compared with 1
under batched greedy search. Sharper targets are not sufficient for better play.

Level3-2 has five greedy deaths: x=2192,2198,2193,3138,3138. This is evidence
of two observed failure regions rather than the earlier single-failure claim
of one clustered location. Sampling's 29/30 versus 25/30 is promising but has
limited failure counts and needs confirmation (paired rescues 5, losses 1).

Level8-1 dies in every one of the 150 attempts across the five controllers;
there are no timeouts. Each individual condition is 0/30, with a Wilson upper
95% endpoint of 11.4%. These results do not establish zero possible completion
probability, but none of the tested controller changes provides a successful
baseline. The cause of its weak competence remains undiagnosed.

## Policy target agreement and state information

Each controller's target probe samples 15 episodes, 24 correlated states per
episode (360 states). They are fresh evaluation-search targets, not historical
training replay. The search uses the trained prior, so agreement is partly
self-referential and cannot establish correct action values.

Greedy-condition results:

| Level | KL(search target || policy), nats | Prior maximum probability | Search top-tie fraction | Within-episode shuffle CE penalty, nats |
|---|---:|---:|---:|---:|
| Level1-1 | .0386 | .2537 | 85.0% | .8382 |
| Level6-1 control | .0267 | .2503 | 86.1% | 1.5222 |
| Level3-2 | .0673 | .2434 | 70.8% | 1.0084 |
| Level8-1 | .0485 | .2526 | 82.5% | 1.8678 |

The head closely tracks current batched-search targets. Shuffling state/prior
pairs worsens agreement on all four levels, including Level8-1, showing state
information about those targets. This contradicts a globally state-independent
head, but permits local failures and does not establish historical target fit.
The full Level1-1 sample overturns the smoke sample's near-zero shuffle penalty:
the smoke result was just eight states from one predominantly stalled episode.

High tie frequency also occurs on the successful control. Ties are therefore
not sufficient to explain failure. Small probability changes can nevertheless
change greedy choices, which is why distribution closeness and argmax agreement
must be interpreted separately.

Re-search uses the same 64 observations per level, balanced across controller
conditions. The greedy-source subset contains 13 states per level. At 50
simulations, mean maximum target share for batch 4 versus batch 1 is:

| Level | Batch 4 | Batch 1 |
|---|---:|---:|
| Level1-1 | .2600 | .4938 |
| Level6-1 control | .2585 | .4292 |
| Level3-2 | .2600 | .4523 |
| Level8-1 | .2708 | .5323 |

Batching flattens these targets; the completion table shows that making them
sharper is not a general solution for these existing checkpoints.

Repeated noisy search uses eight states and five draws per state. Mean KL to
individual targets versus KL to their finite-sample mean is .151/.075 for 1-1,
.276/.187 for 6-1, .205/.111 for 3-2, and .476/.338 for 8-1. Some disagreement
is attributable to search randomness. Eight states and five repeats do not
support a broad ranking of optimization quality across these levels.

## BatchNorm and the gradient hypothesis

BatchNorm probes use 15 episodes per condition, 24 states per episode, and
three batch regroupings. They compare inference on unchanged model copies,
using either prediction-network batch statistics or batch statistics throughout
the network. They do not test completion under modified normalization.

On greedy-source observations, substituting prediction-network batch statistics
changes the policy's top action on 30.0% of Level1-1 states and 41.8% of
Level6-1 states, averaged over regroupings. Mean moved probability mass (TV)
is similar: .0266 versus .0275. Fresh-target KL increases by .0102 and .0170
respectively. Whole-network substitutions increase KL by .0509 and .0701.
Both substitutions also increase greedy-source target KL on 3-2 and 8-1.

Thus normalization sensitivity is real, but the successful control shares it
and substituting these statistics does not improve greedy-source target fit.
This does not establish BatchNorm as the cause, rule out other calibration
interventions, or measure whether imagined states dominate running statistics.
The batches contain real observations, not the learner's real/imagined mixture.

No gradient-magnitude or gradient-conflict experiment was run. Source inspection
shows one backward/optimizer step per batch and a policy objective of root loss
plus the mean of five recurrent losses. Equal nominal weights do not guarantee
equal gradient influence. The existing .5 recurrent-hidden-state gradient hook
acts upstream into dynamics/earlier latents, not on direct prediction-head
parameter gradients. Dynamics-dominated learning remains an untested hypothesis.

## Recommended next tests

Freeze sampling-only (batch 4, temperature .25, root epsilon 0) as the Level1-1
candidate and confirm it against greedy on the reserved trials, retaining both
conditions on Level6-1 to measure the trade-off. A level-specific controller
choice must be declared as part of the evaluated system. No confirmation jobs
were submitted as part of reading these results.

For a causal explanation of the 1-1 stall, examine saved failure traces and
compare alternative action sequences from matched emulator states. For the
gradient hypothesis, separately measure real-state and averaged imagined-state
policy gradients on shared parameters, including their norms and alignment.
Those would be new diagnostic measurements on fresh data, not a reconstruction
of unavailable historical replay.

Source reports for each level are `development/<Level>/summary.json`,
`policy_targets.json`, and `policy_batchnorm.json`; paired results and failure
locations come from the corresponding `episode_XXXX.json` files.
