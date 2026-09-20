# Authoritative handoff status — 20 September 2026

This page supersedes submission-time status in older protocol notes and the September 19 handoff. Scientific inputs are frozen in `evidence.json` (performance/commitment) and `diagnostic_evidence_20260920.json` (completed follow-ups). Do not treat a planned experiment as a result.

| Experiment | State at this snapshot | Result / interpretation | Evidence |
|---|---|---|---|
| Specialists and human comparison | Complete for selected 21-level benchmark | 203/630 greedy, 202/630 sampled; 5-3 remains eligible but not evaluated here | `images/human_vs_agent_runthrough_all_levels_stochastic.json` |
| Three-level gated confirmation | Complete and audited | 1-1: 11→86/100; 6-1: 84→91/100; 1-3: 100→100/100 | [Results](../STALL_SAMPLING_CONFIRMATION_RESULTS_V1.md) |
| Failure branches | Complete and audited | Local jump/release interventions distinguish signatures; survival is not completion | [Results](../FAILURE_BRANCHES_RESULTS_V1.md) |
| Value/commitment | Complete and audited | Holds rescue selected states but harm successful references; imagined/observed gaps vary | [Results](../VALUE_COMMITMENT_RESULTS_V1.md) |
| Actual search-leaf substitution | Complete: 1,120 new branches, 56 roots; metadata recovered and checked | Eight-decision substitution rescues all selected 6-1 pit roots, not obstacle roots; harms some 1-1 references | [Results](../SEARCH_LEAF_VALUES_RESULTS_V1.md) |
| Reconstructed value-target audit | Complete: 2,800 existing records, 56 roots | Position-eight online-bootstrap proxy summaries available; original training replay and lagged target network are unavailable | [Protocol](../VALUE_TARGET_AUDIT_V1.md), `diagnostic_evidence_20260920.json` |
| All-level gated confirmation | Not run | Protocol/seed reservation and incomplete 5-3 evaluation needed | [Next experiments](NEXT_EXPERIMENTS.md) |
| T7 multi-level comparison | Training complete; adequate per-level confirmation outstanding | Historical pooled training rates are not deployment evidence | [Report](REPORT.md) |
| T8 distilled student | No completed student result established | Audit teacher coverage before selecting a student experiment | [Next experiments](NEXT_EXPERIMENTS.md) |

Value-target jobs 6722603 (all 21 array elements), 6722604 (all 35 elements) and audit 6722606 were checked through `sacct`: all COMPLETED, exit 0:0. The summary has `complete=true`, 2,800 records and 56 roots. This resolves the older “submitted” status. It does not reconstruct unavailable historical learner targets.

The search-leaf metadata serialization bug was repaired separately from outcome auditing. All 56 short recovery replays matched the saved traces using recorded caches. No full gameplay outcomes were replaced. The original failed validation/report artifacts remain preserved.

Current handoff validation, environment verification and packaging results are recorded in [REPRODUCE.md](REPRODUCE.md) and `verification/`. Personal access and ownership remain recipient-specific; see [ACCESS_AND_OWNERSHIP.md](ACCESS_AND_OWNERSHIP.md).
