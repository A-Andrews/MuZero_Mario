# Resuming MuZero–Mario

Start at [START_HERE.md](START_HERE.md) for the presentation and verification; read [REPORT.md](REPORT.md) for the scientific account. Scientific evidence is frozen as of 20 September 2026; these operational instructions were updated on 22 September. The documentation handoff was committed in `5ce9143`, and the experimental implementation and remaining project documentation were committed in `312b577` ("Diagnose failures and prepare for pause"). Preserve the frozen experiment directories and external assets as well as the Git history. See [RELEASE.md](RELEASE.md) for the relationship between commits and packaged snapshots.

## Read in this order

1. [Scientific report](REPORT.md): human context, achievements, current rates, diagnosis and remaining work.
2. [Independent controller confirmation](../STALL_SAMPLING_CONFIRMATION_RESULTS_V1.md): the 900-episode practical improvement.
3. [Value and action commitment](../VALUE_COMMITMENT_RESULTS_V1.md): the newest completed mechanism evidence.
4. [Search-leaf results](../SEARCH_LEAF_VALUES_RESULTS_V1.md) and [current status](STATUS.md): both leaf substitution and reconstructed value-target audits are complete.
5. [BACKLOG.md](../../BACKLOG.md) and [CLAUDE.md](../../CLAUDE.md): historical recipes and machine-specific details. Read dated updates carefully; older “pending” or “paused” prose may be superseded.

## Render the handoff without a GPU

From the repository root, using an environment with NumPy and Matplotlib:

```bash
.venv/bin/python scripts/plot_project_handoff.py
```

The original five-figure renderer reads only [evidence.json](evidence.json). It checks scope, completion denominators, missingness and diagnostic-root counts, then writes five PNGs plus vector PDF/SVG versions, a CSV and provenance JSON to `images/handoff/`. No emulator, torch, ROM, network or cluster output mount is needed. Paths resolve relative to the script, so it can also run from another working directory. A temporary Matplotlib cache avoids writing user configuration directories.

The original five figures answer these questions; two new diagnostic figures and the talk are built by `scripts/build_handoff_presentation.py`:

| Figure | Question |
|---|---|
| [01 human context](../../images/handoff/01_human_context.pdf) | Where does each specialist stand relative to recorded humans, in reliability and successful speed? |
| [02 failure balance](../../images/handoff/02_failure_balance.pdf) | Does sampling solve failure or change its type? |
| [03 controller confirmation](../../images/handoff/03_confirmed_controller.pdf) | Which controller benefit replicated on new seeds? |
| [04 failure mechanisms](../../images/handoff/04_failure_mechanisms.pdf) | What do action holds and imagined/observed value comparisons actually establish? |
| [05 inside MuZero](../../images/handoff/05_inside_muzero.pdf) | How do observation, learned prediction, search and action selection connect? |

For a short presentation, use 05 → 01 → 02 → 03 → 04, then the report's remaining-work table/list. The existing [merged action-distribution slide](../../images/action_policy_search/probabilities_all_levels_merged.pdf) adds behavioural detail.

The evidence snapshot contains selected fields from the source files below, plus their SHA-256 hashes. It is a frozen summary, not a live scan. Do not silently replace it with an updated cohort. New data should create a dated snapshot and label development/confirmation scope explicitly. Counts in the commitment heatmap are recomputed from individual roots, not transcribed from prose. Source hashes identify the read inputs; they do not re-audit every historical trace.

## Data and experiment map

`outputs/` is a symlink to `/projects/u6oz/atdandrews/MuZero_Mario/outputs`; frozen metadata may use the equivalent `/lus/lfs1aip2/projects/...` path. The portable report works without that mount; reproducing raw analysis requires the underlying data.

| Location | Contents / authority |
|---|---|
| `images/human_vs_agent_runthrough_all_levels_stochastic.json` | Completed 21-level development comparison, trial records and frozen-checkpoint provenance |
| `outputs/human_trajectories/human_level_stats.json` | Cached human repetition/first-life counts and successful durations; five participants |
| `outputs/controller_policy/stall-sampling-confirmation-v1-20260911/` | Three-level confirmation, manifest, summaries, individual traces and audit |
| `outputs/controller_policy/failure-branches-v1b-20260916/` | Completed 4,620 conditional branches; v1b supersedes the original root-selection setup |
| `outputs/controller_policy/value-commitment-v1-20260918/` | Completed branches, values, audit/summary and `evidence_summary.json` used here |
| `outputs/controller_policy/search-leaf-values-v1b-20260919/` | Completed 56-root follow-up with audited outcomes and recovered search metadata; original v1 failed validation |
| `diagnostic_outputs/value-target-audit-v1b-20260920/` | Completed 2,800-record target audit; shared home output after project inode exhaustion |
| `outputs/controller_policy/followup-v1-20260911/` | Earlier two-arm confirmation and policy-gradient probes; not the later three-arm seed set |
| `outputs/runs/` | Training runs, resolved configurations, `best.pt` / `latest.pt` and metadata |
| `outputs/specialist_trajectories/`, `outputs/specialist_trajectories_v2/` | Teacher corpora; audit actual successful coverage before student training |
| `/projects/u6oz/atdandrews/MuZero_Mario/exports/` | Model and comparison ZIP bundles; manifests identify exact contents |
| `images/action_policy_search/` | Human/prior/search and executed-action figures with CSV/JSON and uncertainty |
| `logs/` | Scheduler stdout/stderr; a submission or passing smoke is not proof of a full result |

Historical bundles: `muzero_mario_models_20260905.zip` (23 checkpoints, including incomplete 5-3 and out-of-active-scope 2-2) and `muzero_mario_comparison_20260905.zip` (8 models). Consult each manifest for training provenance and hashes; those historical bundles do not imply all their models are competent or all are active comparison levels. The September bundle includes `frames_to_obs`; older bundle documentation may not.

## Resuming training

The exported model bundles support inference and evaluation; their optimizer, scheduler and RNG state were removed. They cannot be passed directly to the current training resume loader. Use an original `outputs/runs/<run>/checkpoints/latest.pt` for continuation, retaining its symlink target, saved configuration and launcher recipe. The [training restart guide](TRAINING_RESUME.md) explains checkpoint selection, configuration, output isolation and budget checks; the [original-checkpoint inventory](verification/training-checkpoints-20260922.json) identifies the 29 runs represented in the two bundles.

Original training checkpoints restore weights, optimizer state, saved scheduler state, step counters and saved RNG state. Self-play replay, worker/emulator state and the historical lagged target network are not restored. Treat continuation as a restart with retained learner state, not an exact uninterrupted trajectory. No training restart was executed for this documentation update.

## Code map and conventions

| Component | Start reading here |
|---|---|
| Environment, action set and observations | `src/env/env.py`, `src/env/preprocess.py`, `src/env/mario_actions.py` |
| Representation, dynamics and prediction | `src/muzero/networks.py` |
| Tree search, node backup and selection | `src/muzero/mcts.py`, `node.py`, `utils_mcts.py` |
| Replay, targets and optimization | `src/muzero/buffer.py`, `targets.py`, `muzero.py` |
| Workers, GPU inference and coordination | `src/selfplay/worker.py`, `inference_server.py`, `coordinator.py` |
| Human conversion and scoring | `scripts/convert_human_bk2.py`, `src/muzero/human_data.py`, `human_baseline.py` |
| Existing checkpoint evaluation | `scripts/eval_human_benchmark.py`, `eval_controller_diagnostic.py` |
| Failure interventions | `scripts/diagnose_failure_branches.py`, `diagnose_value_commitment.py`, `diagnose_search_leaf_values.py` |
| Specialist recipe | `scripts/submit_specialist.sh`; resolves overrides beyond `conf/muzero.yaml` |
| Portable model use | `scripts/_bundle_README.md`, `scripts/_bundle_load_model.py` |

`best.pt` is selected by noisy self-play completion, not deployment reliability. Never assume it is the best greedy checkpoint. Construct networks from checkpoint `cfg_snapshot`, use eval mode for frozen inference, and preserve precision/batching/reset settings for exact comparisons. Root noise, sampling temperature and leaf batch are separate controls.

Model input is float in [0,1]; stored observations are uint8. The standard specialist input has four channels at 96×96, and medium hidden states are 192×6×6. `frames_to_obs` expects raw RGB at native 60 Hz. A 30 fps video silently changes temporal scale. The converted human corpus does not currently provide the complete original-frame indexing needed for internal TR alignment; deterministic BK2 replay can recover it. Use the bundle helper for collaborator-supplied frames.

## Resume from the completed diagnostics

Both the search-leaf intervention and value-target audit are complete. Read [STATUS.md](STATUS.md), the report's new follow-up sections and [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md). Search-leaf outcomes and metadata recovery passed; the initial serialization error affected reporting, not gameplay. The value-target audit's 2,800 records are in `diagnostic_outputs/value-target-audit-v1b-20260920/`, outside the project-output symlink because of inode exhaustion. Do not resubmit either completed experiment as if it were pending.

The [protocol](../SEARCH_LEAF_VALUES_V1.md) specifies unchanged production search, exact observation-only equivalence, actual emulator replay of selected search paths, one/eight-decision value replacement, successful controls, and final completeness/trace-hash checks. Read current scheduler state before resubmitting to avoid duplicate work. The separate handoff check performs focused tests, exact trace replay for clips, and a small development-seed model evaluation; it does not train models. See [REPRODUCE.md](REPRODUCE.md).

The intervention has privileged emulator information. Replacing an imagined leaf value with the same network's value on an actual observation does not make that value ground truth. Terminal knowledge introduces an additional difference; inspect terminal-touching paths separately. Changed rankings, local survival and full completion are distinct endpoints.

## Verification and storage

Use focused tests for the component being changed. Existing project notes report that full pytest on a login node can fail in multiprocessing/inference tests and leave a large core dump; full training/emulator verification belongs in a compute allocation. Always scope collection to `tests/`. The handoff renderer only needs a lightweight CPU run and visual inspection of exports.

Do not run training from current defaults to reproduce an old specialist: use its saved configuration and original launcher recipe. Preserve frozen checkpoints, manifests and source snapshots before edits. Existing history includes both byte-quota and file-count-quota failures; inspect available space/inodes before large arrays. No data cleanup or job launch is part of this handoff.
