# Controller investigation: completed results, 2026-09-11

Slurm audit: confirmation jobs 6457937/6457938, local intervention jobs
6459327/6459328, all 18 tasks of array 6459440, and figure jobs 6458425/6459441
completed with exit code 0. All corresponding full summaries have `complete=true`.
No infrastructure errors are recorded in confirmation or coverage evaluations.
This report separates held-out confirmation, development coverage, and local
intervention endpoints. They must not be pooled into one performance estimate.

## Reserved-seed confirmation

Each cell uses all 100 predeclared environment/search seed pairs, unchanged
checkpoints, 50 simulations and a 2000-decision limit. Sampling means temperature
.25, no root noise, leaf batch 4. These seeds were reserved before controller
selection on the 30-trial development experiment.

| Level | Greedy completed | Sampling completed | Greedy timeouts / deaths | Sampling timeouts / deaths |
|---|---:|---:|---:|---:|
| 1-1 | 10/100 | 79/100 | 74 / 16 | 0 / 21 |
| 6-1 control | 90/100 | 68/100 | 5 / 5 | 0 / 32 |

Wilson 95% intervals: 1-1 greedy 5.5–17.4%, sampling 70.0–85.8%; 6-1 greedy
82.6–94.5%, sampling 58.3–76.3%. Paired trial outcomes: sampling rescues 73
greedy failures and loses 4 greedy successes on 1-1; rescues 9 and loses 31 on
6-1. Thus the strong 1-1 improvement and adverse control effect both replicate.
The development 28/30 sampling rate was not a stable 93% performance estimate:
the independent confirmation estimate is 79/100. Neither result establishes
human-level play or robustness across independently trained checkpoints.

## Full 22-level development comparison

Exactly 30 attempts per controller per level, including all deaths and timeouts.
The four original completed levels are reused; the other 18 contribute 1,080
new episodes. New greedy and sampled runs share checkpoint copies and seeds.
The current checkpoint training steps match those recorded in the older
five-trial figure; the older figure did not retain checkpoint hashes.

| Level | Greedy | Sampled |
|---|---:|---:|
| 1-1 | 4/30 | 28/30 |
| 1-2 | 0/30 | 11/30 |
| 1-3 | 30/30 | 17/30 |
| 2-1 | 2/30 | 7/30 |
| 2-2 | 0/30 | 3/30 |
| 2-3 | 14/30 | 5/30 |
| 3-1 | 0/30 | 4/30 |
| 3-2 | 25/30 | 29/30 |
| 3-3 | 26/30 | 19/30 |
| 4-1 | 12/30 | 14/30 |
| 4-2 | 0/30 | 1/30 |
| 4-3 | 0/30 | 0/30 |
| 5-1 | 15/30 | 13/30 |
| 5-2 | 5/30 | 3/30 |
| 6-1 control | 28/30 | 19/30 |
| 6-2 | 0/30 | 2/30 |
| 6-3 | 21/30 | 14/30 |
| 7-1 | 8/30 | 5/30 |
| 7-3 | 8/30 | 2/30 |
| 8-1 | 0/30 | 0/30 |
| 8-2 | 5/30 | 9/30 |
| 8-3 | 0/30 | 0/30 |

Descriptive equal-level totals: greedy **203/660** completions, **98** timeouts,
**359** deaths; sampling **205/660** completions, **2** timeouts, **453** deaths.
Sampling has more completions on 10 levels, fewer on 9, and the same on 3.
It produces at least one completion on 19 levels versus 14 for greedy. These
counts show increased coverage of levels with any success, not a meaningful
overall reliability gain. Individual small differences remain uncertain at n=30;
the 18-level extension is development data, not independent confirmation.

Examples support different failure mechanisms: 1-2 improves from 0 to 11
completions while 21 timeouts disappear; 4-1 loses all 18 timeouts but gains
16 deaths, with completions only 12→14. Levels 1-3, 2-3, 3-3, and 6-3 suffer
more deaths under sampling. Levels 4-3, 8-1 and 8-3 remain 0/30 under each arm,
all deaths. Their lack of competence under these controllers remains unexplained;
it should not be attributed to a stall mechanism by analogy.

## Local Level1-1 mechanisms

All 19 original development greedy timeouts qualified under the predeclared
stall rule. Five eight-decision prefixes were evaluated from each exactly
replayed root, then the unchanged greedy controller resumed with a shared fixed
continuation seed. Horizon: 128 decisions total. Escape: end at least 64 pixels
forward without death, or complete within the horizon. No branch completed the
level within this short horizon, so escape counts are not completion counts.

| Prefix | 1-1 escapes / 19 | 1-1 deaths | 6-1 escapes / 19 | 6-1 deaths |
|---|---:|---:|---:|---:|
| Recorded next eight actions, reference | 0 | 0 | 13 | 0 |
| Release all buttons | 2 | 0 | 9 | 5 |
| Right+jump | 9 | 1 | 13 | 4 |
| Right+run+jump | 9 | 1 | 16 | 1 |
| Left four, then right+run+jump four | 9 | 0 | 16 | 1 |

The paired control episodes originally completed 17/19; two original failures
were retained. Control roots match fractional elapsed time, not terrain across
levels. The intervention branches within each level start from identical replayed
observations and action histories. A local rescue establishes sufficiency at a
tested root, not a universally safe pulse or a unique internal network cause.

The Level1-1 roots separate descriptively into two groups:

**Earlier staircase, x=2354–2402 (17 roots):** every recorded reference prefix
is eight plain-right actions. The reference escapes 0/17; right+jump escapes
9/17 with one death; releasing alone escapes 0/17. Some unsuccessful jumps make
32–48 pixels of progress, below the declared escape threshold, consistent with
moving partway up the staircase and stopping again. This is evidence that jump
initiation/sequencing can release some stalls; one short pulse is not sufficient
for every state.

At all 17 roots, maximum visits are tied and the lowest-index winner is plain
right. Ties are between right and right+jump (9 roots), right and right+run (7),
or right and right+run+jump (1). The policy prior's largest probability is
right+jump on **15/17** roots. Thus at those roots the policy head already ranks
a jump action highest, while the search/action-selection path executes plain
right. This supports investigating search discretization/tie handling rather
than concluding that the head learned no action preference. It does not isolate
tie-breaking as the sole cause: seven maximum-visit ties contain no jump action,
and the successful control also has frequent ties.

**Later staircase, x=2930/2946 (2 roots):** the reference repeatedly holds
right+jump. Adding more jump input escapes 0/2, while releasing buttons escapes
2/2; left then right+run+jump also escapes 2/2. This is consistent with a need to
release and re-press jump, rather than simply increase jump frequency. It is an
inference from only two roots, not a general causal conclusion about jump-button
handling. Screenshots identify the late staircase in these examples.

## Implications for the policy-learning hypothesis

The earlier full probes showed state information and close agreement between
the head and fresh batched search targets, including on 6-1. The gradient probes
found no recurrent-policy domination specific to 1-1: median recurrent/root head
gradient ratios in batch-stat mode are .812 on greedy 1-1 trajectories and
1.219 on the successful 6-1 control. Those measurements exclude other objectives,
optimizer state and historical replay. BatchNorm sensitivity remains shared by
the control and has not been established as a failure cause.

Together, the evidence currently favors local action selection and action timing
as an explanation for a substantial part of 1-1's stalls. The upstream reasons
for flat search and weak action margins may still matter, but the results do
not justify assuming global policy-head failure or changing BatchNorm first.

## Next experiment suggested by these results

Test a controller that preserves greedy behavior during progress and introduces
short, seeded exploration only after a predeclared low-progress trigger. Compare
it against greedy and always-sampled controllers, retain 6-1 and another strong
level such as 1-3, and record both rescues and introduced deaths. First test
trigger/continuation behavior on development data; freeze any selected rule
before assigning a new untouched confirmation seed set. The existing 100
confirmation pairs have now been used and cannot validate a newly tuned rule.

A separate tie-breaking diagnostic could compare deterministic lowest-index,
prior-based tie resolution, and seeded uniform resolution of exact visit ties.
It would isolate one selection mechanism, but cannot on its own address the
seven early ties lacking jump actions or establish the two late release cases.
Neither next experiment was submitted as part of this results audit.

Updated figures: `images/human_vs_agent_runthrough_all_levels_stochastic.pdf`
(22-level development) and `images/human_vs_agent_runthrough_updated_confirmation.pdf`
(100-trial confirmation for 1-1/6-1, labeled development for 3-2/8-1).
