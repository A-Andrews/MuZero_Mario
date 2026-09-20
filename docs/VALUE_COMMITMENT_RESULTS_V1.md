# Value and commitment results, 2026-09-19

**Complete:** both arrays 6680517/6680518 and final audit 6680519 finished 0:0.
All 1,680 new commitment branches and all 56 value roots passed the audit.
Compute analysis job 6702124 produced `evidence_summary.json` under
`outputs/controller_policy/value-commitment-v1-20260918/` using
`scripts/summarize_value_commitment.py`. It supersedes the first descriptive
summary from 6702119 by separating value statistics by failure signature and
including representative action sequences. No new training or controller
deployment occurred. The previous 18 tests and six replay smoke checks passed.

The sample is 23 failed source episodes (46 roots) and five successful source
episodes (10 unique reference roots). Each root has ten paired branch seeds.
Repeated references were deduplicated; distinct source episodes were retained
even when their observed trajectories look identical. Counts below are
conditional branches, not independent level starts or population success rates.
Hold-1 is the reused normal controller, with the first 16 decisions reproduced
for every root/seed. Durations 2/4/8 hold its own first selected action, then
resume the original controller. All episode limits remain unchanged.

## Greedy-source action commitment

| Selected state | Source episodes / roots | Completions: hold 1 | Hold 2 | Hold 4 | Hold 8 |
|---|---:|---:|---:|---:|---:|
| 6-1 obstacle, late (decision 222) | 6 / 6 | 0/60 | 0/60 | 0/60 | 60/60 |
| 6-1 obstacle, early (206) | 6 / 6 | 0/60 | 0/60 | 0/60 | 0/60 |
| 6-1 pit approach, early (270) | 3 / 3 | 0/30 | 30/30 | 30/30 | 30/30 |
| 6-1 pit approach, late (286) | 3 / 3 | 0/30 | 0/30 | 0/30 | 0/30 |
| 6-1 successful references | 2 / 3 | 30/30 | 30/30 | 30/30 | 20/30 |
| 1-1 pipe deaths | 9 / 18 | 0/180 | 0/180 | 0/180 | 0/180 |
| 1-1 later deaths, early (256) | 2 / 2 | 0/20 | 0/20 | 0/20 | 20/20 |
| 1-1 later deaths, late (272) | 2 / 2 | 0/20 | 0/20 | 0/20 | 0/20 |
| 1-1 successful references | 3 / 7 | 70/70 | 70/70 | 40/70 | 30/70 |

At the late 6-1 obstacle, search starts with right+run+jump. The representative
baseline sequence is `[4,5,5,1,4,1,3,4,...]`: right+run+jump, then jump alone,
jump alone, then right. Holding the initially selected action eight decisions
rescues all six episode roots in every paired seed. Holds of two or four still
die. This is direct evidence that subsequent action changes can spoil an
initially useful action. It does not show that search planned an eight-action
sequence, nor isolate why each subsequent choice was poor.

At the earlier pit approach, the representative sequence begins
`[2,0,2,5,...]`: right+jump followed by NOOP. Holding right+jump for just two
decisions changes all three source roots from death to completion. At the later
root the initial action is plain right; holding it does not complete. Hold 8
turns those deaths into timeouts. At the early obstacle root the initial action
is also plain right; hold 4 turns deaths into timeouts without completion.

The successful 6-1 reference at decision 260 provides an essential counterexample:
normal control completes 10/10, while holding its first right+jump action for
eight decisions causes 10/10 deaths. All other 6-1 reference holds complete.
Thus blanket commitment is not justified. On 1-1, hold 8 also loses 40/70
successful-reference branches. At the pipe failures, initial actions are right
or right+run, and all tested holds still die. The previous deliberate jump
prefixes escaped those local deaths; longer execution of the selected action
does not substitute for selecting an appropriate action.

## Stall-sampled source states, kept separate

- Post-rescue episode 11: normal continuation dies 10/10 at both roots. Hold 8
  completes 10/10 at both; hold 4 completes only at the later root.
- Post-rescue episode 59: fresh normal continuations complete 10/10 at both
  roots. Hold 8 loses one at the later root; holds 2/4 preserve all successes.
- Lost-success episode 45: at decision 577 normal continuation completes 4/10,
  holds 2/4 complete 10/10, and hold 8 completes 6/10. The hold-8 arm rescues
  all six baseline failures but loses all four baseline successes at that root.
  At the earlier root, hold 8 loses one of ten baseline successes.

These are the same ten paired search streams. Sampled and greedy controller
evidence is not pooled, and a net gain does not hide lost paired successes.

## Imagined-value consistency

The following numbers are mean absolute differences between values after
imagined prefixes and values after encoding the corresponding actual emulator
observations. They are in reward/value units, not percentages. Four fixed
prefixes per root are equally weighted; seeds do not create new predictions.
"Reanchored" means one recurrent step starting from the encoded actual previous
observation, not eight successive imagined steps.

| Root group | Roots × prefixes | Gap at h=1 | Gap at h=8 | Reanchored gap at h=8 |
|---|---:|---:|---:|---:|
| 1-1 pipe deaths | 18 × 4 | 22.9 | 77.8 | 12.3 |
| 1-1 later deaths | 4 × 4 | 1.9 | 11.8 | 12.7 |
| 1-1 successful references | 7 × 4 | 1.7 | 18.0 | 4.5 |
| 6-1 obstacle deaths | 12 × 4 | 0.6 | 16.6 | 11.9 |
| 6-1 pit deaths | 6 × 4 | 0.5 | 6.2 | 2.1 |
| 6-1 successful references | 3 × 4 | 0.7 | 12.5 | 2.9 |

The pipe group has substantial multi-step inconsistency: reanchoring reduces
the eight-step gap for 68/72 root/prefix pairs. For example, at episode 1,
decision 163, a held right+jump prefix receives imagined endpoint value **48.6**,
actual-observation value **183.5**, and reanchored one-step value **180.6**.
For the logged dying prefix, these are **165.5**, **13.6** and **101.4**: even the
one-step correction leaves a substantial error on this transition. Immediate
discounted reward errors are under 0.1 in magnitude for all four prefixes at
this root. Large errors here are in values, not short-prefix reward prediction.

The successful references prevent overgeneralization: imagined/observed gaps
also occur there, and successful 6-1 references have an average eight-step gap
comparable to some failure groups. The discrepancy is not itself a universal
failure detector. Nor does reanchoring always help: it worsens the aggregate
eight-step gap at the later 1-1 death states.

## Real-observation value errors remain

Replacing imagined values with real-observation values cannot automatically
solve everything. At the late 6-1 obstacle root (episode 12, decision 222), the
held right+run+jump prefix has imagined endpoint value **4.4**, real-observation
value **18.0**, and a realized full discounted return of **161.2**. Of the roughly
−156.7 full-return discrepancy, −1.8 is prefix reward error, −13.5 the discounted
imagined/observed value gap, and −141.4 the observed-state value residual.
This is compatible with serious undervaluation of this successful continuation,
but its target differs from the network's training-policy value target.

The later 1-1 death at decision 272 is another example: the logged prefix has
nearly matching imagined/observed endpoint values (132.4/131.9), yet its learned
return is 138.6 versus realized return 11.2. Good latent-value agreement is
therefore insufficient for accurate continuation evaluation.

These comparisons use the specified prefix and continuation under the original
2,000-step cap; they are not unbiased errors against an infinite-horizon or
training-policy target. See the protocol for the exact decomposition.

## Diagnosis and next discriminating test

We have causal evidence for action-sequence sensitivity at particular 6-1 and
1-1 states, and strong imagined/observed value inconsistency at the 1-1 pipes.
We have not shown that correcting imagined values changes the actual MCTS
decision: the fixed eight-action diagnostic prefixes need not be the paths the
50-simulation search explores. This remains the next link to test.

A focused next test should log the actual searched paths/depths and replace
only leaf values with values computed from the emulator observations reached
by those exact paths, retaining original policy priors, predicted rewards and
search budget. Use paired searches and successful 6-1 controls, and distinguish
changed action rankings from improved outcomes. This privileged emulator-based
test would be a diagnostic, not a deployment proposal. An unchanged bad choice
would leave value-head/representation error and search allocation unresolved.
Global action holds or BatchNorm changes are not supported as fixes by this round.
