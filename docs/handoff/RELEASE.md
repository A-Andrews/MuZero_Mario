# Frozen handoff release

The current operational revision is **muzero-mario-handoff-20260922**. Scientific results, figures and presentation retain the **20 September evidence snapshot**; this revision clarifies training resumption and Git provenance. Local artifacts are in `handoff_release/20260922/` at the repository root:

- `muzero_mario_source_20260922.zip`: source, configurations, tests, documentation, evidence snapshots and figures; includes the training restart guide/inventory and `HANDOFF_SOURCE_MANIFEST.json` with per-file SHA-256 hashes and Git base/dirty status.
- `muzero_mario_presentation_20260922.zip`: unchanged HTML/PDF talk, slide source, speaker notes and six MP4 clips with provenance.
- `release.json` and `SHA256SUMS`: archive identities.

## Git and archive provenance

The original documentation commit is `5ce9143`. The experimental implementation and remaining project documentation were subsequently committed in `312b577` ("Diagnose failures and prepare for pause"). The September 20 archive was created against `5ce9143` with those experimental changes present in its working-tree snapshot; it is not a clean export of that commit. At the September 22 review, its 307 recorded files matched the `312b577` working tree except `.gitignore`.

The September 22 source archive records `312b577` as its Git base and includes the current documentation and packaging changes. The **archive's per-file hashes** identify the captured files even before this revision has a Git commit. Its recorded Git status distinguishes these changes from the existing `mario.stimuli` submodule file-type/permission changes; the integration assets remain external.

The original **muzero-mario-handoff-20260920** archives and their `release.json`, `source_manifest.json` and `SHA256SUMS` remain unchanged directly under `handoff_release/`. The historical verification records under `docs/handoff/verification/` retain their original dates and scope; `training-checkpoints-20260922.json` is the new presence/configuration inventory, not a training restart test. The September 22 archive verification report is stored alongside that release as `verification.json`.

Regenerate deliberately, after checking that the desired source/results are final:

```bash
.venv/bin/python scripts/package_handoff.py --date 20260922 --out handoff_release/20260922
cd handoff_release/20260922
sha256sum -c SHA256SUMS
```

The source package deliberately excludes the ROM, raw human recordings, checkpoint payloads, installed environment, output directories and MP4s. The presentation package contains the MP4s. External model bundles and experiment locations are recorded in `verification/assets.json` and HANDOFF.md. A fresh extraction can render the figures/talk without those external model/data assets; models need access to them.

Do not silently overwrite a published version after new experiments. Supply a new `--date YYYYMMDD` and a separate `--out` directory for a later release. Nothing is uploaded, shared or sent by packaging.
