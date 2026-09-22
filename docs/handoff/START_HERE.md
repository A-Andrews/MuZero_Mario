# Start here: MuZero–Mario handoff

**For a group presentation:** open [presentation/index.html](presentation/index.html) locally. Arrow keys move slides; **N** shows speaker notes. The [PDF](presentation/MuZero_Mario_handoff.pdf) is the static fallback; [speaker notes](presentation/SPEAKER_NOTES.md) provide an approximately 18-minute narrative. Keep the `clips/` folder beside the HTML for videos.

**For the scientific account:** read [REPORT.md](REPORT.md), then the [current experiment registry](STATUS.md). The main outcome is partial specialist competence, a confirmed targeted controller improvement, and evidence for several distinct failure mechanisms. Broad human-level performance has not been established.

**For a new contributor:** follow [REPRODUCE.md](REPRODUCE.md), use [HANDOFF.md](HANDOFF.md) to navigate the implementation and assets, and read the three [next-experiment cards](NEXT_EXPERIMENTS.md). Resolve the recipient-specific items in [ACCESS_AND_OWNERSHIP.md](ACCESS_AND_OWNERSHIP.md).

**For training continuation:** read [TRAINING_RESUME.md](TRAINING_RESUME.md). Exported models are for inference; resumption needs original training checkpoints and starts with fresh self-play replay. Operational documentation was revised on 22 September; the scientific evidence remains the 20 September snapshot.

## What is in the handoff

- Scientific report with citations, denominators, scope and limitations.
- Seven numbered figures, exported as PNG/PDF/SVG, and machine-readable numerical evidence.
- Thirteen presentation slides, speaker notes and six trace-verified gameplay clips with selection/provenance records.
- A lightweight plotting environment specification plus a separate inventory of the working cluster environment.
- Model/evaluation verification records, an external-asset inventory, and source/presentation bundles with checksums.
- A prioritized research plan and explicit outstanding access/ownership decisions.

The human-covered scope is 22 levels. The broad current performance figure has 21 model levels; 5-3 is retained as missing. The archived 23-model bundle also contains historical 2-2, which is outside the active comparison scope. Model files and raw human data are referenced rather than duplicated into the small presentation bundle.

[Release contents and checksums](RELEASE.md) describe the named source/presentation bundles.

## First session with the successor

1. Give the talk and agree what the evidence does and does not establish.
2. Have the successor regenerate a figure and read the manifest for one checkpoint.
3. Verify their own data and cluster access, then run the bounded evaluation check.
4. Select one next-experiment card, assign an owner and record a review date.

The current account's successful checks are evidence that the workflow works here. Recipient access and a person independently following the instructions still need direct confirmation.
