"""Tiny-config CPU smoke test. Mirrors train_muzero.py but with tiny hyperparams
so the full pipeline exercises in a few minutes on one CPU core.
"""
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import subprocess


def main():
    overrides = [
        "device=cpu",
        "env.levels=[Level1-1]",
        "selfplay.num_workers=1",
        "selfplay.max_queue_size=4",
        "selfplay.max_trajectory_length=200",
        "mcts.num_simulations=8",
        "training.batch_size=16",
        "training.min_replay_transitions=100",
        "training.save_every_train_steps=20",
        "training.log_every_train_steps=5",
        "training.weight_broadcast_every=10",
        "training.total_env_steps=1000",
        "muzero.unroll_K=3",
        "muzero.n_step=3",
        "model.hidden_channels=32",
        "model.dyn_blocks=2",
        "model.pred_blocks=1",
        "model.rep_blocks=[1,1,1,1]",
        "wandb.mode=disabled",
    ]
    cmd = [sys.executable, str(_REPO_ROOT / "scripts" / "train_muzero.py")] + overrides
    print("[smoke] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
