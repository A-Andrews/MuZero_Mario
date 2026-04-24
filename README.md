# muzero_mario

MuZero on Super Mario Bros (NES), sibling to [ppo_study](../ppo_study). Uses the same `gym-retro` + `mario.stimuli` integration for apples-to-apples comparison with the PPO baseline.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`mario.stimuli/` is a symlink to the `ppo_study` copy (ROM + state files).

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
