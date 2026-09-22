# Resuming training from the saved models

Operational review: **22 September 2026**, against code committed in `312b577`. This guide describes the current loader; it does not claim that a new training restart has been tested.

## Choose the right checkpoint

| Artifact | Intended use | Training resume through the current loader |
|---|---|---|
| Exported `checkpoints/Level*.pt` and comparison-bundle models | Frozen inference, evaluation and activations | Unsupported: optimizer, scheduler and RNG state were removed by `scripts/package_models.py` |
| Original `outputs/runs/<run>/checkpoints/latest.pt` | Continue from the latest saved learner state | Supported by the loader, subject to compatible configuration and environment |
| Original `outputs/runs/<run>/checkpoints/best.pt` | Return to a selected earlier learner state | Same loader, but rewinds to that checkpoint's counters; self-play selection does not establish best evaluation performance |

The [checkpoint inventory](verification/training-checkpoints-20260922.json) maps the 31 exported models to 29 distinct original run directories. All 29 `latest.pt` files and saved Hydra configurations were readable at review time. It records resolved targets and file sizes, plus hashes of the small configuration files. This is a presence/readability inventory, **not a checkpoint-content audit, backup or recipient-access test**. The export hashes in `assets.json` identify stripped model files and must not be used as hashes of the original training checkpoints.

For example, the Level6-1 original continuation point is `outputs/runs/spec-level6-1/checkpoints/latest.pt`; the exported evaluation model is derived from that run's `best.pt`. These are different selections. Use each bundle manifest's `source_checkpoint` to identify its original selection, and the inventory's `latest` entry to locate the continuation point.

## What the loader restores

`src/checkpoint.py` saves the online network, optimizer, optional scheduler, `training_step`, `env_step`, `cfg_snapshot`, and Python/NumPy/learner PyTorch RNG states. `MuzeroLearner.load()` in `src/muzero/muzero.py` restores the learner state and saved RNG states.

The following are reconstructed at startup rather than restored:

- Self-play replay trajectories and priorities: `scripts/train_muzero.py` creates a fresh `TrajectoryBuffer`. Configured human data is loaded separately for imitation runs.
- The lagged target network and its synchronization history: when reanalysis is enabled, the first target refresh initializes it from the restored online network.
- Worker/emulator trajectories, worker RNG progression and in-flight self-play work.

Consequently, this is a **restart with retained learner state**, not an exact continuation of an uninterrupted training trajectory. Record the restart boundary in subsequent analysis. Historical replay and target-network state cannot be recovered from these checkpoints; an inference-only export also cannot recover optimizer history.

The loader **does not apply `cfg_snapshot` automatically**. It loads into a network, optimizer and scheduler constructed from the supplied configuration. Match the original configuration, including model architecture, preprocessing/action set, imitation inputs, replay/target settings and schedule definitions, then record intentional changes. A successfully loaded weight tensor alone does not establish an equivalent training setup.

## Before starting a continuation

1. Select the original run using the inventory and bundle manifest. Preserve the original checkpoint and its configuration before allowing a launcher to write into that run directory. `latest.pt` is often a relative symlink to `step_<N>.pt`: retain the target file too, or make a dereferenced copy. Checkpoint rotation can remove old `step_*.pt` files from an active run.
2. Retain `.hydra/config.yaml`, `.hydra/overrides.yaml`, the checkpoint's `cfg_snapshot`, original launcher recipe and code version. The current defaults and generic specialist launcher are not a substitute for an imitation or curriculum run's settings. Preserve these outside any output directory that will be reused; Hydra can rewrite its metadata.
3. Prefer a new run name/output directory for a new experiment or an extension of a completed frozen run. Supply an absolute `resume.path` pointing to the preserved **full training checkpoint**. Keep the original results untouched and record the parent checkpoint path/hash in the new experiment manifest. The usual chain launcher sets `run_name` and `hydra.run.dir` from its first argument.
4. For continuation in an existing run directory, the same non-null `run_name` lets `scripts/train_muzero.py` find `checkpoints/latest.pt` automatically. An explicit `resume.path` takes precedence. Check scheduler state first so two jobs do not write to the same run. Reusing the run name also reuses the W&B identity.
5. Check both the checkpoint's `env_step` and any `TRAINING_COMPLETE` marker. `training.total_env_steps` is the **total cumulative budget**, not additional steps: a checkpoint at 15M needs a budget of 20M for a further 5M. An already-reached budget exits without training. Record an intentional extension rather than deleting completion evidence; using a new output directory avoids carrying the old sentinel into an extension.
6. Before allocating a long chain, compare its resolved configuration with the saved one and perform a bounded restart on a compute node. Confirm the expected resume path/counters, replay refill, learner updates and a newly written full checkpoint. This training check remains to be performed; the handoff's existing GPU smoke verifies evaluation only.

For planning, `MUZERO_CHAIN_DRY_RUN=1` makes `scripts/submit_chain.sh` print its submission plan without submitting jobs. It does not validate the checkpoint or effective training configuration, and its remaining-budget estimate reads `latest.pt` in the destination run; for a new destination with an explicit parent checkpoint, account for the parent's progress separately.

The original checkpoints remain in external project storage. This documentation revision does not copy them into the small source archive or establish a backup. Access and storage stewardship remain on the separate [ownership checklist](ACCESS_AND_OWNERSHIP.md).
