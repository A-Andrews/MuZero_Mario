# Completed failure-branch results, 2026-09-18

All **4,620/4,620** branches and all 47 cases are complete. Resumed control array
6607749 and remaining array 6607753 completed 0:0; final aggregation and trace
hash audit 6607758 completed 0:0. The earlier quota interruption did not replace
any trial or seed. Read-only evidence summary job 6668070 produced
`outputs/controller_policy/failure-branches-v1b-20260916/evidence_summary.json`.
The experiment's `summary.json` and per-case `full/case_*/summary.json` remain
the detailed outcome/prediction sources. No new model training was performed.

The sample is **23 failed source episodes**, with two probe points each,
matched references drawn from **five distinct successful episodes**, and a
separate trigger comparison. There are ten unique successful-reference roots;
the matching scheme reuses them across cases. Ten branch seeds at the same
state are conditional repetitions, not ten independent gameplay starts.

## 1-1 early deaths: survival and completion remain different

For all nine pipe-death episodes at x=898, right+jump and right+run+jump prefixes
avoid death at both tested timings in all ten branch seeds. Two source episodes
(indices 32 and 69) then complete in 10/10 for each jump arm/root. The other
seven episodes always time out under the subsequent greedy controller.
This extends the interim observation: jumping reliably avoids this local death
in the selected cases, but only some resulting trajectories complete.

For the two later early-death episodes (indices 16 and 82, original deaths at
x=1948), releasing the buttons at the later root (decision 272, x=1821)
completes in 10/10 per episode; continuing dies in 10/10. Right+jump at that
same root still dies, while right+run+jump times out. Releasing at the earlier
root also times out. Position, action sequence and timing all matter.

## 6-1 control: timing and action duration matter even for a strong model

Six failed 6-1 episodes share the x≈1394 approach. At the earlier probe
(decision 206), jumping avoids the immediate death but every branch eventually
dies. At the later probe (222), both jump prefixes complete in 10/10 per source
episode, whereas continuing dies in 10/10. At that later root the unforced
controller already selects right+run+jump as its first action. Forcing that
same action for eight decisions succeeds: first-action ranking alone cannot
explain the failure. Subsequent action selection is material.

For the three 6-1 pit-death episodes, release or right+run+jump completes in
10/10 at either root; continuation dies in 10/10. Right+jump succeeds at the
earlier root but times out at the later one. These are selected failure states,
not an estimate of a deployable controller's completion rate.

The successful references rule out a blanket intervention rule. For example,
the intact 6-1 reference at decision 260 completes in 10/10 when continued, but
forcing right+jump there causes 10/10 deaths. At another successful 6-1 root,
right+run+jump causes timeouts. Their nearby x coordinates do not make these
states identical: speed, enemy phase and jump phase can differ. Flat priors or
tied search visits also occur in these successful references.

## Failures following stall sampling

For the post-rescue death at episode 11, continuing or replaying the logged
prefix dies in 10/10 at both roots. Release and both jump prefixes each complete
in 10/10. This confirms avoidable action-sequence/timing failure locally.

For post-rescue episode 59, fresh sampled continuations complete in 10/10 at
both roots. Forcing the later logged eight-action prefix instead recreates
10/10 deaths, while release/right+jump completes in 10/10 and right+run+jump
in 9/10. The original sampled sequence is consequential; this is not a state
from which every continuation must fail.

For the single lost greedy success (episode 45), the direct comparison before
its first trigger yields **10/10 greedy and 10/10 gated completions** under
the ten new branch streams. It does not reproduce the original rare loss.
Later in that failed sampled trajectory, at decision 577, fresh gated
continuation completes 4/10, replaying the logged prefix 0/10, and each forced
alternative 10/10. At the earlier root, continuation already completes 10/10.
This supports sensitivity to the sampled sequence and subsequent trajectory;
it does not establish that the original trigger was intrinsically wrong.
Fresh branch RNG streams are not restoration of the original search RNG state.

## What the policy and learned predictions show

At one pipe root (episode 1, decision 147), the policy favors right+jump
(21.3%) over right (18.0%), yet search chooses right from a right/right+run
visit tie. The action with the highest policy prior is therefore not always
the source of the bad decision.

At that root, the learned eight-action return estimate is about **166** for
the logged sequence and **114** for held right+jump. Actual discounted returns
under the specified continuation are about **−2** and **121**, respectively
(the latter still times out). The model's endpoint value after imagined jump
actions is about **115**, compared with **183** when its representation/value
network receives the actual endpoint observation. At the later pipe root,
the logged sequence has an imagined endpoint value around **165**, while the
actual endpoint observation receives about **14**.

These are concrete prediction inconsistencies worth following up. They do not
isolate dynamics from representation/value-head behavior or prove BatchNorm
causation. Realized return follows the specified continuation; learned value
targets may reflect a different policy. Eight-action prefixes also cannot be
treated as single-action ground truth. At the 6-1 timing-sensitive root, search
has a unique winning visit action and already chooses the action that succeeds
when held, so a universal explanation based solely on final visit ties fails.

## Implications

The evidence supports at least three components: local obstacle handling,
action timing/duration, and inaccurate or inconsistent imagined values. An
initial death can be avoided yet expose the already-known later greedy stall.
The statement that the policy head simply failed to learn is too broad: it
sometimes proposes a useful action that search rejects, and sometimes the
first selected action is already appropriate but later decisions spoil it.

The next focused diagnostic should separate learned-prediction error from
action-sequence instability at these exact states, retaining successful 6-1
references. No controller modification or new performance run was submitted
during this results review. The human-data scope remains unchanged: 2-2/7-2
excluded; 5-3 eligible. No claim about untested failure states on other levels
follows from these branches.
