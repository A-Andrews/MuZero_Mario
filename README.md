# muzero_mario

MuZero on Super Mario Bros (NES), sibling to [ppo_study](../ppo_study). Uses the same `gym-retro` + `mario.stimuli` integration for apples-to-apples comparison with the PPO baseline.

## Results and internship handoff (20 September 2026)

Start with [START_HERE](docs/handoff/START_HERE.md) for the presentation, verification and next experiments. Read the [project report](docs/handoff/REPORT.md) for achievements, human-performance context, MuZero's internals, evidence about failures and remaining experiments. The [handoff guide](docs/handoff/HANDOFF.md) maps code, data, checkpoints and how to resume work.

Seven [figures](images/handoff/) are available as PNG, PDF and SVG, with a portable [evidence snapshot](docs/handoff/evidence.json). Regenerate them without model inference:

```bash
.venv/bin/python scripts/plot_project_handoff.py
.venv/bin/python scripts/build_handoff_presentation.py
```

The active scope is 22 human-covered levels, with 21 evaluated in the current model benchmark and 5-3 explicitly missing. Broad human-level performance is not established. Independent confirmation shows a stall-triggered controller improves 1-1 from 11/100 to 86/100 completions, with unchanged weights and additional RAM-based stall detection; see the report for controls and limits.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`mario.stimuli/` supplies the ROM integration and state files. It is a direct clone in this workspace; the original BMRC setup used a symlink to the `ppo_study` copy.

## Train

Local smoke test:
```bash
python scripts/smoke_test.py
```

Full training on BMRC:
```bash
sbatch scripts/submit_bmrc.sh
```

Hydra overrides:
```bash
python scripts/train_muzero.py env.levels='[Level1-1]' selfplay.num_workers=1 mcts.num_simulations=10
```

## Layout

- `src/env/` — retro + mario.stimuli env wrappers (ported from ppo_study)
- `src/muzero/` — networks, MCTS, buffer, learner
- `src/selfplay/` — parallel self-play workers + coordinator + replay eval
- `src/logs/` — wandb + mp4 video helpers
- `conf/` — Hydra configs
- `scripts/` — entrypoints + SLURM
- `tests/` — unit tests

## Logging

wandb is required. Run `wandb login` once before training.
- per-step loss metrics
- per-episode self-play returns per level
- at every checkpoint: mp4 replay rollouts per level uploaded as `wandb.Video`

## Reference

Structural reference: [Muzero-Hanoi](/well/costa/users/zqa082/Muzero-Hanoi). This repo does not depend on it; it was used as a pattern source.
