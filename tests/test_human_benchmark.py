"""Checkpoint selection + provenance for the human run-through benchmark job."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.eval_human_benchmark import _repo_rel, pick_run  # noqa: E402


def _mk_run(root, name, rate=None, step=None, ckpt="best.pt"):
    d = root / name / "checkpoints"
    d.mkdir(parents=True)
    if ckpt:
        (d / ckpt).write_bytes(b"x")
    if rate is not None:
        (d / "best.json").write_text(json.dumps(
            {"completion_rate_100ep": rate, "training_step": step, "env_step": 1}))
    return root / name


def test_picks_the_higher_scoring_of_two_runs_for_a_level(tmp_path):
    """A level can have both a T6 self-play run and a T7 imitation rescue —
    whichever actually got further should win, not a hard-coded family."""
    _mk_run(tmp_path, "spec-level1-3", rate=0.02, step=100)
    _mk_run(tmp_path, "spec-imit-level1-3", rate=0.85, step=552654)
    rate, run_dir, ckpt, meta = pick_run("Level1-3", tmp_path, "best.pt")
    assert run_dir.name == "spec-imit-level1-3"
    assert rate == pytest.approx(0.85)
    assert meta["training_step"] == 552654


def test_self_play_run_wins_when_it_scored_higher(tmp_path):
    _mk_run(tmp_path, "spec-level3-2", rate=0.99, step=348830)
    _mk_run(tmp_path, "spec-imit-level3-2", rate=0.10, step=5)
    _, run_dir, _, _ = pick_run("Level3-2", tmp_path, "best.pt")
    assert run_dir.name == "spec-level3-2"


def test_run_without_the_checkpoint_is_skipped(tmp_path):
    """Levels that never completed have no best.pt at all."""
    _mk_run(tmp_path, "spec-level1-3", rate=None, step=None, ckpt=None)
    _mk_run(tmp_path, "spec-imit-level1-3", rate=0.85, step=1)
    _, run_dir, _, _ = pick_run("Level1-3", tmp_path, "best.pt")
    assert run_dir.name == "spec-imit-level1-3"


def test_no_candidates_returns_none(tmp_path):
    assert pick_run("Level7-2", tmp_path, "best.pt") is None


def test_level_tag_does_not_bleed_across_levels(tmp_path):
    """'Level1-1' must not also match a spec-level1-10 style directory."""
    _mk_run(tmp_path, "spec-level1-1", rate=0.5, step=1)
    _mk_run(tmp_path, "spec-level1-2", rate=0.9, step=2)
    _, run_dir, _, _ = pick_run("Level1-1", tmp_path, "best.pt")
    assert run_dir.name == "spec-level1-1"


def test_missing_best_json_still_yields_a_candidate(tmp_path):
    _mk_run(tmp_path, "spec-level2-2", rate=None, step=None)
    rate, run_dir, ckpt, meta = pick_run("Level2-2", tmp_path, "best.pt")
    assert run_dir.name == "spec-level2-2" and rate == -1.0 and meta == {}


def test_repo_rel_handles_relative_absolute_and_outside(tmp_path):
    repo = Path(__file__).resolve().parent.parent
    assert _repo_rel(Path("outputs/runs/x/checkpoints/best.pt")) == \
        "outputs/runs/x/checkpoints/best.pt"
    assert _repo_rel(repo / "scripts" / "eval_human_benchmark.py") == \
        "scripts/eval_human_benchmark.py"
    outside = tmp_path / "elsewhere.pt"          # not under the repo: kept verbatim
    assert _repo_rel(outside) == str(outside)
