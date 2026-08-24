# images/

Generated figures, tracked in git so a result can be pointed at from a commit.

| File | Produced by | What it shows |
|------|-------------|---------------|
| `human_vs_agent_runthrough.pdf` | `sbatch scripts/submit_human_benchmark.sh` | Each level's **best** checkpoint played through the level a few times, against the distribution of successful human run-throughs. Every row names the run and training step it came from. |
| `human_vs_agent_runthrough.json` | same | Per-rollout results plus provenance (run, checkpoint path, training/env step, git sha, seeds). |

The PDF is vector and a few tens of KB — safe to commit. Regenerating overwrites
in place, so `git diff --stat images/` shows when a figure moved.
