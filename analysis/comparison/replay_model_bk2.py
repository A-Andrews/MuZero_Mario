"""Build a model_progress JSON by replaying a run's saved greedy-replay .bk2 set.

Each training run records one `Level{W}-{S}.bk2` per level at every replay-eval
checkpoint (outputs/runs/<run>/videos/step_<N>/). Those are the agent's own
greedy-MCTS rollouts, so replaying them reproduces the agent's behaviour at that
checkpoint exactly — no need to re-run MCTS. We read max-x / score / completion
straight from the emulator, the same way we process the human .bk2 files.

    python replay_model_bk2.py <int_path> <videos/step_N dir> <out.json>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_bk2 import replay_bk2  # noqa: E402


def main():
    int_path, bk2_dir, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    m = re.search(r"step_(\d+)", str(bk2_dir))
    step = int(m.group(1)) if m else None
    levels = {}
    for f in sorted(bk2_dir.glob("Level*.bk2")):
        lv = f.stem  # "Level2-1"
        try:
            rec = replay_bk2(str(f), int_path)
            levels[lv] = {"completed": rec["completed"], "max_x": rec["max_x"],
                          "score": rec["score"], "n_steps": rec["n_frames"]}
        except Exception as e:  # noqa: BLE001
            levels[lv] = {"error": str(e)[:200]}
        print(f"{lv}: {levels[lv]}", flush=True)
    data = {"checkpoint": str(bk2_dir), "step": step, "levels": levels}
    Path(out).write_text(json.dumps(data, indent=2))
    print(f"wrote {out}  (step {step})")


if __name__ == "__main__":
    main()
