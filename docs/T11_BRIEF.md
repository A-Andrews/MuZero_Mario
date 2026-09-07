# T11 brief: why the agents don't reliably complete levels

Instructions for an agent picking this up. Read this, then BACKLOG.md's T11 and
T10 sections. Don't re-derive what's below.

## The problem

12 of 22 levels never finish greedily off `best.pt` (`images/human_vs_agent_runthrough.json`),
while the same checkpoints score 0.2–0.99 in noisy self-play. The gap *is* the
problem: headline completion rates depend on MCTS root-Dirichlet noise that
deployment doesn't have.

## Do this first: stop treating it as one failure

It's three, split by failure mode and spread of `final_x`:

| signature | levels | evidence |
|---|---|---|
| hard deadlock | 1-1 | 5/5 hit the 2000-step cap, never died, x spread **16** |
| clustered death | 3-2, 6-3, 4-3 | all failures die at one spot — spread 0, 0, 531 |
| scattered death | 8-1, 8-2, 2-3 | die everywhere — spread 3786, 3151, 2394 |

Level1-1 never dies; Level8-1 never times out. **Any explanation covering both
is wrong.** Say which signature you're working on.

## Leading hypothesis — test the links, don't assume the chain

`mcts_visit_max_frac` sits at **0.23–0.27 for entire runs, across all 12 T6
specialists**. The most-visited root action never exceeds ~a quarter of
simulations. That visit distribution *is* the policy training target.

> flat search → flat policy target → weak policy head → greedy argmaxes a near-tie

T10 supports the last link independently: lesioning the policy head's **1,266
parameters** (0.006% of the net) is catastrophic on every level tested, while
the 7.0M-parameter forward model is less so. Greedy eval leans almost entirely
on the policy head.

## Already established — do not redo

- **Discount fix** (0.997→0.999, `completion_bonus` 100→200) is what unlocked
  completions at all; every pre-fix run was ≤4%.
- **Pit-gap local optimum** (T6 Level1-3/4-3): agents die at a gap, `mcts_root_q_mean`
  flat all run. "Walk to the edge and stop" is an optimum the value head correctly
  certifies. Human demos rescued 1-3 (→0.85), barely 4-3 (→0.01).
- **More training doesn't fix it**: Level5-3 got 15M steps self-play + 15M with
  imitation, zero completions, dies at x≈907 every rollout.
- **Not a curriculum artefact**: both T7 curriculum arms complete 1/60 and 0/60
  greedy rollouts across 12 levels.
- **Greedy is a knife-edge per checkpoint**, not a property of the policy — see
  BACKLOG's "open problem" section.

## Tools that already exist

- `scripts/diag_policy_collapse.py` — separates "head collapsed as a function"
  from "head merely confident" (`argmax_switch_rate`); `visit_vs_prior_tv` asks
  whether search adds anything over the prior; `--tail-window` reads the stall.
- `scripts/diag_search_sharpness.py` — sweeps `leaf_batch` × `num_simulations`
  to separate mechanical causes of flat visits from a genuinely flat head.
- `scripts/lesion_eval.py` — component ablations; `--sims` sweeps search depth.
- `scripts/eval_human_benchmark.py --run <run>` — score one model across levels.

Jobs 6385358 (policy collapse) and 6385359 (sharpness) were launched 2026-09-07
on one level per signature. Results: `outputs/diag_policy/`, `outputs/diag_sharpness/`.

## Rules

1. **Always include Level6-1 as a control.** Same recipe, same budget, completes
   5/5. Whatever you find on a failing level must *not* hold there, or it isn't
   the explanation.
2. **Check n before believing anything.** T10 reversed twice when n went up:
   "value lesions are free" was one level at n=4, and an intact baseline of n=3
   made a vacuous test look interpretable. Intact baselines are the denominator.
3. **A model that can't complete a level intact can't serve as a baseline.**
   Level8-2/5-2 lesion runs were unmeasurable for this reason.
4. **Report greedy and noisy separately.** Never quote a self-play rate as if it
   were deployed performance.
5. **Compute nodes only** (`sbatch`), outputs under `/projects`, never home —
   see CLAUDE.md.

## Deliverable

Which signature has which cause, with the Level6-1 control ruling out the
alternatives. A fix is secondary to a correct diagnosis.
