import json

import pytest

from scripts.plot_updated_runthrough import arm_summary, build_report, load_level


def test_failed_attempts_count_in_rate_but_not_success_time():
    result = arm_summary([{"completed": True, "steps": 500},
                          {"completed": False, "steps": 2000, "timed_out": True}])
    assert result["n"] == 2 and result["rate"] == .5
    assert result["successful_steps"] == [500] and result["successful_median"] == 500
    assert result["timeouts"] == 1
    failed = arm_summary([{"completed": False, "steps": 100}])
    assert failed["successful_median"] is None and failed["rate"] == 0


def test_incomplete_confirmation_cannot_appear_in_figure(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    folder = tmp_path / "confirmation" / "Level1-1"
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps({"complete": False, "split": "confirmation"}))
    with pytest.raises(ValueError, match="incomplete"):
        load_level(tmp_path, "Level1-1", "confirmation")


def test_infrastructure_failure_cannot_be_treated_as_gameplay_failure():
    with pytest.raises(ValueError):
        arm_summary([{"status": "error", "completed": False, "steps": 0}])


def test_full_overview_excludes_levels_without_human_gameplay(tmp_path, monkeypatch):
    from argparse import Namespace
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    path = tmp_path / "reference.json"
    path.write_text(json.dumps({"generated": "earlier", "human": {}, "levels": [
        {"level": "Level2-2", "checkpoint": "old.pt", "training_step": 10, "run": "specialist",
         "rollouts": [{"completed": False, "steps": 100}]}]}))
    report = build_report(Namespace(reference=path, all_levels=True, confirmation=None, development=None))
    assert report["levels"] == []
    assert report["excluded_without_human_data"] == ["Level2-2"]


def test_scope_keeps_human_levels_without_successful_time_band(tmp_path):
    from scripts.human_level_scope import human_levels
    (tmp_path / "sub-01_level-w5l3_rep-000_seg0.npz").touch()
    assert human_levels(tmp_path) == ["Level5-3"]
    with pytest.raises(ValueError, match="No converted"):
        human_levels(tmp_path / "missing")


def test_extension_replaces_older_rows_with_completed_new_data(tmp_path, monkeypatch):
    from argparse import Namespace
    import scripts.plot_updated_runthrough as plotting
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    ref = tmp_path / "reference.json"
    ref.write_text(json.dumps({"generated": "earlier", "human": {}, "levels": [{"level": "Level1-2"}]}))
    extension = tmp_path / "extension"
    extension.mkdir()
    (extension / "manifest.json").write_text(json.dumps({"evaluation_levels": ["Level1-2"]}))
    seen = []
    def load(folder, level, split):
        seen.append((folder, level, split))
        return {"level": level, "split": split, "arms": {}}
    monkeypatch.setattr(plotting, "load_level", load)
    report = build_report(Namespace(reference=ref, all_levels=True, confirmation=None,
                                    development=None, extension=extension))
    assert seen == [(extension, "Level1-2", "development")]
    assert report["levels"][0]["split"] == "development"
