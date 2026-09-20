# Failure branches: interrupted run and exact resumption

Superseded by completed results: `docs/FAILURE_BRANCHES_RESULTS_V1.md`.
All 4,620 branches and final audit finished successfully after the resumption below.

Original array **6602494** completed cases 0–4; cases 5–46 failed with
`OSError: [Errno 122] Disk quota exceeded`. Dependent audit 6602502 was cancelled.
This is an infrastructure interruption, not a model outcome. The shared Lustre
project 1483806624 has a **51,200,000-file hard limit**. Current quota inspection
showed about 23,000 free file slots despite substantial free disk capacity.

Compute preflight **6607690** verified **567 saved branch records and their
trace hashes**, checked frozen script/case/plan identities, and passed a synced
16 MiB write test. The last summary files contained only 566 branches; one
additional valid result had been written before its summary update failed.
Five complete cases contain 500 branches. The other saved results are partial.
No 6-1 control case finished in the interrupted array; no final diagnostic
conclusion should be drawn from this subset.

## Conditional findings from completed cases

Cases 0 and 4 are two early-death source episodes at the 1-1 pipe. At each of
two roots, continuing the original controller dies in 10/10 branch seeds.
Right+jump and right+run+jump prefixes avoid death in 10/10 each, but **all
subsequently time out**, so these are not level-completion rescues.

Case 2 is one post-rescue death trajectory. At both roots, continuing or
replaying the next eight logged actions dies in 10/10. Each alternative prefix
(release, right+jump, right+run+jump) instead completes in 10/10. This is evidence
for an avoidable action/timing error at this trajectory, not proof of which
network component causes it or of general performance across levels.

Successful reference case 1 keeps 10/10 completions when continued, but its
jump prefixes lead to timeouts and its release prefix leads to deaths.
Successful reference case 3 keeps 10/10 completions under continuation or jump
prefixes, while release leads to timeouts. Repeated reference roots represent
the same source state, not independent observations. The branch-seed count is
conditional on each selected state; there are only a few source episodes here.

## Resume jobs

- **6607749**, tasks 28–31: two 6-1 failure/reference pairs first, after successful
  storage/integrity preflight.
- **6607753**, tasks 5–27 and 32–46: remaining unfinished cases after those controls.
- **6607758**: final CPU aggregation and trace audit after both resumed arrays.

Concurrency remains four GPUs. The same frozen source, checkpoints, case plan,
roots, and seeds are used. Completed branches are retained and verified by the
runner; infrastructure failures retry the exact missing trial. No replacement
seeds or new tuning is introduced. The shared file quota may change as other
project jobs write files; the preflight establishes current write access, not
reserved capacity.
