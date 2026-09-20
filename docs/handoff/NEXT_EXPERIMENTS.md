# Three next-experiment cards

These are actionable proposals, not completed results or permission to launch a broad training fleet. Start with one decision. Keep successful controls and report harms, not only net gains. The existing development and confirmation seeds are already consumed.

## 1. Does gated sampling improve reliability broadly?

**Question:** does the unchanged three-level candidate retain its benefit across the human-covered level set?

**Inputs:** corpus-derived 22-level scope; the exact checkpoint SHA for every level; existing stall trigger settings (96 decisions, ≤16 pixels, up to 32 sampled decisions at T=.25); no root noise, 50 simulations, leaf batch four, 2,000-decision cap. Include 5-3's incomplete `latest.pt` explicitly; never call it `best.pt`. Record model training provenance and the candidate's additional RAM inputs.

**Implementation boundary:** the current evaluator's `stall_gated` profile hard-codes levels 1-1/6-1/1-3. Extend it under a new profile with tests preserving the frozen old protocol. Do not simply pass more levels to the existing validation. Start from `scripts/eval_controller_diagnostic.py`, `scripts/stall_sampling_controller.py` and `scripts/human_level_scope.py`.

**Execution sequence:** freeze a new manifest and fresh environment/search seed pairs; validate one old development replay plus a new-profile smoke; run paired greedy/gated trials; audit all planned records and trace hashes. A proposed budget is 100 pairs × 22 levels × two arms = 4,400 episodes. Time a small development pilot first; estimate total GPU-hours from observed seconds/episode rather than copying old walltime limits.

**Primary endpoint:** equal-level mean change in first-life completion. Also report per-level paired rescues/losses, death/timeout changes, sampled fraction and intervals. Use participant-aware uncertainty for any renewed human comparison; human and model trials are not paired.

**Decision rule:** retain the candidate for further testing only if the predeclared paired aggregate improvement is supported and control harms are acceptable under a tolerance agreed before launch. A provisional conservative tolerance is no greater than five percentage points lost on either established reliable control (1-3/6-1); mark this as a proposal until accepted. A net gain must not conceal a severe level-specific regression. If the result is mixed, keep a conditional or level-specific claim and test any newly tuned controller on another untouched cohort.

**Deliverables:** immutable manifest, per-trial outcomes, paired-difference analysis, coverage figure including 5-3, and a concise decision report. Completion of this experiment alone does not resolve human start/practice mismatch.

## 2. Can the value error be corrected with the existing target rule?

**Question:** do useful observed states need better fitting/coverage, or does the short target itself fail to supply a corrective signal?

**Inputs:** the completed 56-root value-target audit; full post-prefix endpoints and terminal flags; existing checkpoints, held-out successful controls; diagnostic continuation provenance. The online network used in reconstruction is a proxy for the unavailable lagged target net. Historical replay coverage cannot be recovered from these artifacts.

**Evidence motivating the test:** at the selected late 6-1 obstacle endpoint, value ≈18 while the 10-step proxy target is ≈131 and terminal return ≈160. At the early pit example, the 10-step proxy remains below the current value despite later success. A single horizon explanation is therefore insufficient.

**First executable analysis:** rerun `scripts/snapshot_handoff_diagnostics.py` to reproduce the portable 280 root/arm summaries from the audited source files. Group position-eight prediction/target residuals by failure signature, continuation outcome and controller. Aggregate by source episode; ten seeds are not ten independent episodes. Separate true-terminal returns from timeout finite sums, retain successful references and report missing endpoints.

**Controlled pilot after that analysis:** copy a checkpoint into a new output directory. Compare an unchanged baseline with (A) extra updates on selected useful trajectories using the current 10-step rule and (B) the same sampling/update budget with a longer target. Freeze separate training/validation episodes and never evaluate on the exact training roots as proof of generalization. This fine-tuning pilot needs a new runner/protocol; it is not already implemented by the audit script.

**Decision rule:** if the current rule corrects values and improves held-out actions/outcomes, prioritize coverage/fitting. If a longer target adds a reproducible benefit at matched exposure and updates, investigate target design. If neither improves behaviour, revisit policy mismatch, representation and action sequencing. Check retention on successful control levels and replicate promising changes across independent training seeds before making a fleet claim.

**Deliverables:** grouped residual report, a preregistered small pilot with exact update budget, validation split and retention endpoints, then paired behavioural results. Do not treat privileged leaf substitution as a deployable model repair.

## 3. Can one model retain the specialists' skills?

**Question:** does specialist distillation produce a useful multi-level agent beyond the existing curriculum arms?

**Inputs:** manifests of `outputs/specialist_trajectories/` and `_v2/`, successful episodes/steps by level, teacher checkpoint/controller hashes, human/imitation provenance and both 60M-step T7 models. Empty teacher levels remain visible.

**First action:** inventory both corpora rather than assuming v2 solves coverage. Use the loader's actual filename/level filters; inspect trajectory policies and root values. Existing code: `scripts/dump_specialist_trajectories.py`, `src/muzero/human_data.py`, `src/muzero/buffer.py`. All filenames must match the loader's `sub-*.npz` convention. Record sampling temperature/noise used by the teacher.

**Pilot:** choose and freeze a supported subset with a declared minimum successful teacher coverage per level. Use the existing imitation pipeline for student pretraining and RL mixing. Start with a memory estimate from loaded transitions (4×96×96 bytes/observation before other arrays), and validate a small batch before submitting training. The command in BACKLOG's T8 section is historical; adapt its level list to the active human scope and set an explicit budget.

**Comparison:** specialists, student, T7-human and T7-nohuman under the same starts, search settings and life/time budgets. Rerun T7 per-level evaluation with more than one trial. Separate training cost and competence; do not compare noisy self-play peaks to confirmation rates.

**Decision rule:** establish a per-level retention target before training; expand scope only if the student retains skills on the pilot set without hiding unsupported levels. If the student fails mainly off the fixed teacher state distribution, investigate on-policy teacher correction rather than only enlarging the offline corpus.

**Deliverables:** corpus coverage audit, explicit student recipe, cost accounting, per-level matched evaluation and a decision to expand or revise distillation.
