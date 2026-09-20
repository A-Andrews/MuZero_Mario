# Access and ownership sign-off

Current-account read access can be checked automatically. A successor's account permissions and willingness to own the project cannot. No invitations, file shares or messages are sent by this handoff workflow.

| Responsibility | Owner | Required confirmation |
|---|---|---|
| Scientific decisions and interpretation | Not yet named | Can explain the claim/limitations and select the first experiment |
| Code and experiment execution | Not yet named | Can run the setup, checkpoint check and Slurm workflow |
| Model/data storage stewardship | Not yet named | Knows retained locations, quotas, backup and access process |
| Contact while author is away | Not yet named | Agreed contact route and expectations |
| Next review | Not scheduled | Date and concrete output expected |

## Recipient acceptance checklist

- [ ] Has the frozen handoff source/presentation bundle and verifies its SHA-256.
- [ ] Can access the repository and identify the handoff version.
- [ ] Can read the model-bundle manifests and required checkpoints.
- [ ] Has appropriate access to CNeuroMod recordings/converted data for the intended analysis.
- [ ] Has the required Mario integration/ROM and state files; these are external assets, not bundled in the source archive.
- [ ] Can read project output storage and the home `diagnostic_outputs/` audit directory.
- [ ] Has a Slurm account/allocation and a working GPU environment if running evaluations/training.
- [ ] Knows that training uses W&B, while the handoff figures and diagnostic checks do not need a W&B login.
- [ ] Independently regenerates one figure and executes one model check using REPRODUCE.md.
- [ ] Agrees the next-experiment owner, budget, acceptance criterion and review date.

The external-asset inventory in `verification/assets.json` records paths, identities and current-account checks. Machine-specific paths may need remapping on a different cluster. Never put passwords, tokens or private keys in this document; use the institution's existing access process.

## Storage issue to address before large arrays

The latest value-target audit was moved into shared home storage after project-storage inode exhaustion. Its completed results must remain discoverable alongside project outputs. Confirm both byte capacity and inode capacity before further arrays; changing the output directory is a documented workaround, not a repair of the underlying capacity issue. Do not delete old snapshots as part of a routine handover.
