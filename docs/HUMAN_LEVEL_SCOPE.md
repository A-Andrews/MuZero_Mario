# Active level scope: human gameplay required

The user requested exclusion of levels without human data. The converted
CNeuroMod corpus contains 22 levels: worlds 1–8, stages 1–3, except **2-2 and
7-2**. Eligibility is based on converted gameplay files, not successful human
completion or model performance. **5-3 remains eligible** (422 converted
segments); its absence from the existing benchmark is not a lack of human data.

`scripts/human_level_scope.py` derives the list from the corpus filenames.
The default training curriculum and fleet now contain 11 levels after removing
2-2 from their former 12-level subset. Specialist and single-level submission
check human coverage. Human benchmark/replay and updated comparison plots filter
their requested levels using the corpus. The future coverage template is
`docs/controller_coverage_human_v1.json`, with 17 extension levels rather than
18; submit full arrays as 0–16. The earlier template and frozen manifests remain
historical experiment records.

Compute job **6607698** passed 13 targeted tests and regenerated
`images/human_vs_agent_runthrough_updated_all_levels.{pdf,png,json}` and
`images/human_vs_agent_runthrough_all_levels_stochastic.{pdf,png,json}`.
Both now have **21 model/human rows**, explicitly recording 2-2's exclusion in
their JSON metadata. 7-2 was not in those figures. 5-3 remains in the eligible
corpus but has no best.pt in the earlier selected benchmark, so no model result
is invented for it. The four-level action-distribution comparison is unaffected.

No checkpoints, human data, or historical evaluation results were deleted.
The old original benchmark PDF/JSON remains a provenance source; active updated
figures use the new scope. The branch investigation on 1-1 and 6-1 remains
fully within the eligible corpus.
