# Search leaf-value substitution: outcomes, 2026-09-20

All **1,120 new gameplay branches** finished successfully. Validation 6702349
passed all seven smoke states and 22 tests; full arrays 6702399/6702400 completed
0:0. The initial final-report job 6702401 failed because `decisions` was used
for both search diagnostics and rollout length. The length overwrote the
diagnostic list during serialization. This affected reporting, not gameplay.

The repair keeps `decisions` as the rollout length and stores the diagnostic
list under `search_decisions`. Outcome audit **6721905** passed, with 14 focused
tests including a regression test for the collision. It verified every saved
trace hash, root/seed identity, branch length, completion/death/timeout and
source-baseline outcome. First-action changes are recovered directly from the
saved action traces. Compute summary **6721916** produced the results below.
Original frozen runners and original per-seed artifacts remain unchanged.

## What was tested

At every actually explored MCTS leaf, replace only the backed-up value with
the frozen network's value on the actual emulator observation reached by that
path. Imagined hidden states, policy priors, predicted rewards, simulation budget
and checkpoints remain unchanged. Apply this for the next one or eight decisions,
then return to normal search for the remaining original episode allowance.
This uses privileged emulator access and is a diagnostic, not a proposed final
evaluation controller. Real-observation network values are not ground truth.

There are 56 selected roots: 46 from 23 failed source episodes and ten from
five successful source episodes. Every root has ten paired branch seeds. The
denominators below are conditional repetitions, not independent level starts.
No failure group, seed or outcome is excluded after seeing the intervention.

## Greedy-source roots

| Signature | Source episodes / roots | Baseline completions | Substitute 1 decision | Substitute 8 decisions |
|---|---:|---:|---:|---:|
| 6-1 pit approach, early (270) | 3 / 3 | 0/30 | 0/30 | 30/30 |
| 6-1 pit approach, late (286) | 3 / 3 | 0/30 | 0/30 | 30/30 |
| 6-1 obstacle, early (206) | 6 / 6 | 0/60 | 0/60 | 0/60 |
| 6-1 obstacle, late (222) | 6 / 6 | 0/60 | 0/60 | 0/60 |
| 6-1 successful references | 2 / 3 | 30/30 | 30/30 | 30/30 |
| 1-1 pipe deaths | 9 / 18 | 0/180 | 0/180 | 20/180 |
| 1-1 later deaths (1948) | 2 / 4 | 0/40 | 20/40 | 20/40 |
| 1-1 successful references | 3 / 7 | 70/70 | 60/70 | 40/70 |

**6-1 pit failures:** all six early/late roots complete in every paired seed
when substitution continues for eight decisions. Substituting the first decision
alone changes the first action in every branch but still ends in death. Sustained
changes to search valuations can therefore rescue these selected trajectories;
changing only the first choice is insufficient. All three successful 6-1
reference roots retain their successes with either intervention. Unlike the
earlier blanket hold-8 test, this intervention does not kill the successful
pit-reference root at decision 260.

**6-1 obstacle failures:** all remain deaths, including the late decision-222
states where holding the originally selected right+run+jump for eight decisions
previously gave 60/60 completions. Leaf substitution changes the first action
at every late root but does not rescue it. The early root's first action stays
unchanged. This does not isolate the remaining cause, but shows that this
value substitution is insufficient for the obstacle signature.

**1-1 pipes:** substitution for eight decisions produces 20 completions, 90
deaths and 70 timeouts, versus 180 baseline deaths. Both completion gains are
at the earlier roots of episodes 54 and 67; other local survival gains expose
later stalls. One-decision substitution gives zero completions, 160 deaths
and 20 timeouts. First actions change in 80/180 branches in either intervention.
This is partial improvement, not a resolution of the pipe failures.

**1-1 later deaths:** at decision 256, one-decision substitution completes
20/20 while eight-decision substitution times out 20/20. At decision 272,
one-decision substitution still dies 20/20, while eight-decision substitution
completes 20/20. Timing and duration remain critical.

**1-1 successful references:** one-decision substitution loses 10 successes
to timeouts; eight-decision substitution loses 30 (20 timeouts, ten deaths).
Thus preservation of the 6-1 controls does not justify a universal correction
rule across models or states.

## Stall-sampled source roots, kept separate

- Post-rescue episode 11: baseline and one-decision substitution die 10/10
  at each root; eight-decision substitution completes 10/10 at both roots.
- Post-rescue episode 59: baseline and both interventions complete 10/10
  at both roots. The combined post-rescue group changes from 20/40 to 40/40
  with eight-decision substitution, with no lost baseline successes.
- Lost-success episode 45: both interventions preserve 10/10 at the earlier
  root and improve the later root from 4/10 to 10/10. No paired successes lost.

## Metadata recovery and remaining interpretation

Outcome results above are fully audited. Detailed search paths, depths, visit
distributions and matched-state shadow decisions were overwritten in the
original JSON and require reconstruction. The observation-value caches and
complete gameplay traces survived.

`scripts/recover_search_leaf_diagnostics.py` reruns only the original first
16 baseline decisions and first eight intervention decisions, using cached
observation values only. It rejects unrecorded paths and verifies actions,
rewards, positions, player states, terminal flags and Q against original traces.
It writes separate `search_recovery.json` artifacts; no full outcome is rerun
or replaced. Representative recovery array **6721917** and remaining 52-root array
**6721926** completed 0:0. Final summary **6721927** requires and includes all
56 recovered roots. Every short replay matched the original actions, rewards,
states and Q using only recorded cached observations.

Across 4,480 baseline searches, 224,000 leaf backups reached depths 1–5:
35,449 at depth one, 83,735 at two, 96,520 at three, 8,278 at four and 18 at
five. Thus 80.5% reached depth two or three. The one-decision and eight-decision
intervention arms also never exceeded depth five. **No evaluated path reached
an emulator terminal in any arm** (224,000 / 28,000 / 224,000 backups). Terminal
value replacement therefore cannot explain these intervention effects. Backups
are correlated within trees and episodes; these counts are not independent n.
Accounting was checked by compute preflight **6722504**, saved in the next
value-target audit's `preflight.json`.

At all late 6-1 obstacle roots, substitution changes the initial action from
right+run+jump (4) to plain right (1), without rescuing the episode. At the
successful 6-1 pit reference (root 37), the first action stays right+jump and
subsequent changes retain success.

Changing value backups also changes subsequent tree exploration. The causal
result is about this intervention as a whole: it does not distinguish dynamics
from representation/value-head effects, prove BatchNorm causation, or establish
improved whole-level performance. The earlier eight-step imagined-prefix probe
extends beyond normal searched depths here; its gap alone does not quantify the
error encountered by actual search. The matched-leaf intervention does test
values on those actual searched paths.
