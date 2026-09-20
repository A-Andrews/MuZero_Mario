# Imagined values and interrupted action sequences

Requested 2026-09-18. Protocol frozen before these new intervention outcomes.
Completed 2026-09-19: all 1,680 branches and 56 value roots audited; both arrays
and final audit completed 0:0. See [results](VALUE_COMMITMENT_RESULTS_V1.md).
Runner: `scripts/diagnose_value_commitment.py`; template:
`docs/value_commitment_v1.json`; submit wrapper: `scripts/submit_value_commitment.sh`.

## Questions and scope

1. Along the same actual action prefix, how much of the imagined-return error
   comes from predicted rewards, imagined versus observed-state values, and the
   observed-state value's remaining discrepancy from the recorded continuation?
2. Does holding the controller's own first chosen action for 2, 4 or 8 decisions
   change outcomes compared with normal replanning (hold 1)?

Use every failure root and every unique successful-reference root from the
completed `failure-branches-v1b-20260916` experiment: 23 failed episodes,
46 failure roots, and 10 reference roots from five successful episodes.
Deduplicate repeated references by level, source condition, episode and step;
do not merge different episodes merely because they look similar. Exclude the
separate lost-success trigger comparison; its two-controller question differs.
This yields 56 roots, including Level6-1 failures and successful intact controls.
These are selected diagnostic states, not a representative performance sample.
The pipe death, later 1-1 death, 6-1 obstacle/pit deaths and post-rescue failures
remain distinct; early and late roots remain distinct.

## Value probe

Replay each saved logged/release/right+jump/right+run+jump eight-decision prefix
from the original emulator reset, verifying observations at the root and the
actual rewards, positions, terminal flags and player states along the prefix.
Use all ten existing paired continuation seeds, without new outcome selection.
At each horizon h=1..8 record:

- Values, rewards and policy probabilities after unbroken imagined dynamics.
- Value and policy after encoding the actual emulator observation.
- A one-step prediction from the encoded actual preceding observation. Comparing
  this with the unbroken prediction tests accumulation of recurrent error.

With R_hat and R denoting discounted imagined and real prefix rewards, and
G_tail the recorded return after that prefix, the exact arithmetic decomposition is:

`(R_hat + gamma^h V_imagined) - (R + gamma^h G_tail)`

`= (R_hat - R) + gamma^h(V_imagined - V_observed) + gamma^h(V_observed - G_tail)`.

The observed endpoint value is zero on terminal transitions. Save its raw
network output separately. Recorded tails end at the original terminal or
2,000-decision cap, with no bootstrap at the cap. Before h=8 the tail includes
the remainder of the prescribed prefix. Thus this is a policy-conditional,
finite-horizon return comparison, not proof of error against the training value
target. The imagined/observed value gap itself does not depend on that tail.
One-step reanchoring is an offline diagnostic using emulator observations;
it is not a deployable search modification or an isolated BatchNorm test.
Do not equate inconsistent latent predictions with dynamics-only causation:
representation aliasing and value-head sensitivity remain possible.

## Commitment probe

At each root and paired seed, search selects its usual first action. Hold that
same selected action for 2, 4 or 8 decisions, then resume the original greedy
or stall-sampled controller for the remaining original episode allowance.
No action is chosen using the known successful branch. Continue running shadow
search during forced decisions so the RNG stream and instrumentation match the
prior branch protocol; restore and update the stall monitor normally.
Record the proposed and executed actions, first disagreement and first 16 root
search visit/Q diagnostics. Holding 1 is the original controller exactly.

Reuse its audited full saved continuation, verifying unchanged source/model
hashes and reproducing its first 16 decisions for every paired seed at every
root. Stop on disagreement. Ten old seeds are retained across all durations,
not cherry-picked or relabelled as independent starts. There are 1,680 new full
branches (56 roots × 3 new durations × 10 seeds); the 560 baseline comparisons
reuse prior data. Group ten traces in one NPZ per root/duration to limit files.
Greedy-source and sampled-source roots are reported separately.

At the 6-1 decision-222 failure roots, the previous controller already selected
right+run+jump first, while forcing eight such actions succeeded. A duration
effect here directly tests interruption of that initially useful action.
The earlier decision-206 roots and successful 6-1 references test timing and
the damage caused by committing in inappropriate states. First-action holds
may fail at 1-1 roots that initially choose the wrong action; that is informative,
not grounds to substitute an oracle action. A benefit establishes local action
sequence sensitivity, not the quality of an imagined multi-action search plan.

## Execution and interpretation

Validate on compute nodes: arithmetic, pairing and deduplication tests, frozen
source compatibility, then smoke roots spanning both levels and every group.
Full arrays start only after validation passes. Models stay in eval mode; no
training, BatchNorm update, root noise, search-budget change or seed selection.
Completion/death/timeout and local outcomes stay separate. Final aggregation
requires every planned root/duration/seed and validates saved trace hashes.
Report n as source episodes, unique roots and conditional seeds separately.

This round measures the two mechanisms separately. It does not yet show that
repairing values changes search decisions; that would require a subsequent
controlled intervention if these measurements support it.

## Submitted jobs

Output: `outputs/controller_policy/value-commitment-v1-20260918/`.

- **6680504**: completed 0:0 in 2m07s; 18 focused tests and all six representative
  smoke roots passed, including saved-baseline decision checks on both levels.
- **6680517**: Level6-1 roots 35–55, at most two concurrent GPU tasks.
- **6680518**: Level1-1 roots 0–34, at most two concurrent GPU tasks.
- **6680519**: CPU-only final audit and summary, after both full arrays succeed.

Both arrays depend on successful validation; failed dependencies cancel queued
work. Source, checkpoints and plan are frozen under the output directory.
The full arrays run the value probe and commitment branches together per root.
Both arrays and audit are now complete. Compute evidence summary 6702124
separates the failure signatures and checks paired rescues/losses.
