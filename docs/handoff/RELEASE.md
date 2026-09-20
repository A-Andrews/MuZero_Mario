# Frozen handoff release

The release is named **muzero-mario-handoff-20260920**. Local artifacts are in `handoff_release/` at the repository root:

- `muzero_mario_source_20260920.zip`: source, configurations, tests, documentation, evidence snapshots and figures; includes `HANDOFF_SOURCE_MANIFEST.json` with per-file SHA-256 hashes and Git base/dirty status.
- `muzero_mario_presentation_20260920.zip`: HTML/PDF talk, slide source, speaker notes and six MP4 clips with provenance.
- `release.json` and `SHA256SUMS`: archive identities.

The documentation handoff can have its own Git commit while pre-existing experimental implementation remains uncommitted. The **source archive's per-file hashes**, not that documentation commit alone, identify the complete captured working tree. This avoids silently committing unrelated ongoing work while retaining the exact implementation needed by the handoff.

Regenerate deliberately, after checking that the desired source/results are final:

```bash
.venv/bin/python scripts/package_handoff.py
cd handoff_release
sha256sum -c SHA256SUMS
```

The source package deliberately excludes the ROM, raw human recordings, checkpoint payloads, installed environment, output directories and MP4s. The presentation package contains the MP4s. External model bundles and experiment locations are recorded in `verification/assets.json` and HANDOFF.md. A fresh extraction can render the figures/talk without those external model/data assets; models need access to them.

Do not silently overwrite a published version after new experiments. Choose a new dated name in the packager before creating a later release. Nothing is uploaded, shared or sent by packaging.
