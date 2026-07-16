---
name: human-trajectory-corpus
description: Location + provenance of the converted human gameplay corpus and the datalad/git-annex toolchain on Isambard
metadata:
  type: project
---

Converted human imitation corpus (built 2026-07-13): `outputs/human_trajectories/`
(~1.9G) — 8,766 Trajectory .npz segments, 3.25M agent-steps, 771 completions,
22 levels (no w2l2/w7l2/castle levels in the human data). Produced by
`scripts/convert_human_bk2.py` from the datalad clone at `~/data/mario`
(all 3,374 .bk2s downloaded from the anonymous CONP HTTP store).

Toolchain that is NOT in the repo: git-annex standalone arm64 at
`~/tools/git-annex.linux` (must be on PATH for any datalad/annex work);
datalad is pip-installed in the project venv. The user's goal for this data is
imitation learning; integration strategy (buffer seeding vs. fixed human batch
fraction vs. BC pretraining) was still an open design question.
