"""Per-level human reference stats + the leak-free action-eval split."""
import json

import numpy as np
import pytest

from src.muzero.human_baseline import (
    CONVERTER_COMPLETION_BONUS,
    CONVERTER_REWARD_SCALE,
    STATS_CACHE_NAME,
    compute_level_stats,
    load_action_eval_sets,
)
from src.muzero.human_data import select_human_files, split_human_files

A = 12


def _write_seg(path, *, T, level, completed, reward=1.0):
    actions = np.random.randint(0, A, size=T).astype(np.int64)
    policies = np.zeros((T, A), dtype=np.float32)
    policies[np.arange(T), actions] = 1.0
    np.savez_compressed(
        path,
        obs_stacks=np.random.randint(0, 255, size=(T, 4, 8, 8), dtype=np.uint8),
        actions=actions,
        rewards=np.full(T, reward, dtype=np.float32),
        policies=policies,
        root_values=np.zeros(T, dtype=np.float32),
        returns=np.zeros(T, dtype=np.float32),
        priorities=np.ones(T, dtype=np.float32),
        level=level,
        terminal=True,
        completed=completed,
        bk2=path.stem + ".bk2",
    )


@pytest.fixture
def corpus(tmp_path):
    """4 reps of Level1-1: two completed on the first life (100 and 200 steps),
    one completed only on the second life, one never completed. Plus a
    Level2-3 rep so level filtering is exercised."""
    def name(sub, tag, rep, seg):
        return tmp_path / f"{sub}_ses-001_task-mario_level-{tag}_rep-{rep:03d}_seg{seg}.npz"

    _write_seg(name("sub-01", "w1l1", 0, 0), T=100, level="Level1-1", completed=True)
    _write_seg(name("sub-01", "w1l1", 1, 0), T=200, level="Level1-1", completed=True)
    # rep 2: died first (seg0), then finished (seg1) -> completed but not no-death
    _write_seg(name("sub-02", "w1l1", 2, 0), T=50, level="Level1-1", completed=False)
    _write_seg(name("sub-02", "w1l1", 2, 1), T=300, level="Level1-1", completed=True)
    # rep 3: never finished
    _write_seg(name("sub-02", "w1l1", 3, 0), T=40, level="Level1-1", completed=False)
    _write_seg(name("sub-01", "w2l3", 0, 0), T=60, level="Level2-3", completed=False)
    return tmp_path


def test_stats_group_segments_into_reps(corpus):
    s = compute_level_stats(corpus, ["Level1-1"], use_cache=False)["Level1-1"]
    assert s.n_segments == 5 and s.n_reps == 4
    # 3 of 4 reps completed; only 2 did it without dying first
    assert s.n_completed == 3 and s.completion_rate == pytest.approx(0.75)
    assert s.n_no_death == 2 and s.no_death_rate == pytest.approx(0.5)
    # lengths/returns come from the completing segments only (100, 200, 300)
    assert s.length_median == pytest.approx(200.0)
    assert s.length_min == pytest.approx(100.0)
    assert s.return_median == pytest.approx(200.0)  # reward 1.0/step
    assert s.subjects == ["sub-01", "sub-02"]


def test_stats_skip_level_humans_never_played(corpus):
    stats = compute_level_stats(corpus, ["Level1-1", "Level7-2"], use_cache=False)
    assert "Level1-1" in stats and "Level7-2" not in stats


def test_stats_subject_filter(corpus):
    s = compute_level_stats(corpus, ["Level1-1"], subjects=["sub-01"], use_cache=False)["Level1-1"]
    assert s.n_reps == 2 and s.completion_rate == pytest.approx(1.0)


def test_stats_cache_roundtrip(corpus):
    first = compute_level_stats(corpus, ["Level1-1"])["Level1-1"]
    assert (corpus / STATS_CACHE_NAME).is_file()
    # Delete the corpus: a cache hit must not need to re-read any npz.
    for f in corpus.glob("*.npz"):
        f.unlink()
    second = compute_level_stats(corpus, ["Level1-1"])["Level1-1"]
    assert second == first


def test_stats_cache_keyed_by_subject(corpus):
    compute_level_stats(corpus, ["Level1-1"], subjects=["sub-01"])
    blob = json.loads((corpus / STATS_CACHE_NAME).read_text())
    assert "Level1-1|sub-01" in blob["entries"]
    # the all-subjects entry must not be served from the sub-01 scan
    assert compute_level_stats(corpus, ["Level1-1"])["Level1-1"].n_reps == 4


def test_return_at_bonus_rebases_only_the_bonus(corpus):
    s = compute_level_stats(corpus, ["Level1-1"], use_cache=False)["Level1-1"]
    assert s.return_at_bonus(CONVERTER_COMPLETION_BONUS) == pytest.approx(s.return_median)
    expected = s.return_median + (200.0 - CONVERTER_COMPLETION_BONUS) / CONVERTER_REWARD_SCALE
    assert s.return_at_bonus(200.0) == pytest.approx(expected)


def test_action_eval_holdout_never_overlaps_training_files(corpus):
    """The whole point of holdout_only: agreement must not be scored on states
    the imitation loader trained on."""
    levels = ["Level1-1", "Level2-3"]
    train_files, holdout_files = split_human_files(
        corpus, levels=levels, holdout_fraction=0.5, seed=0
    )
    assert holdout_files and not (set(train_files) & set(holdout_files))

    ev = load_action_eval_sets(
        corpus, ["Level1-1"], max_transitions_per_level=10_000,
        holdout_only=True, holdout_fraction=0.5, split_levels=levels, seed=0,
    )
    held_steps = sum(
        len(np.load(f)["actions"]) for f in holdout_files if "_level-w1l1_" in f.name
    )
    assert len(ev["Level1-1"]) == held_steps


def test_action_eval_uses_whole_corpus_when_imitation_off(corpus):
    ev = load_action_eval_sets(
        corpus, ["Level1-1"], max_transitions_per_level=10_000, holdout_only=False
    )
    total = sum(len(np.load(f)["actions"]) for f in select_human_files(corpus, levels=["Level1-1"]))
    assert len(ev["Level1-1"]) == total


def test_action_eval_respects_cap_and_disable(corpus):
    ev = load_action_eval_sets(corpus, ["Level1-1"], max_transitions_per_level=64)
    assert len(ev["Level1-1"]) == 64
    assert ev["Level1-1"].obs.shape[0] == 64
    assert load_action_eval_sets(corpus, ["Level1-1"], max_transitions_per_level=0) == {}


def test_action_eval_skips_unplayed_level(corpus):
    assert load_action_eval_sets(corpus, ["Level7-2"], max_transitions_per_level=64) == {}


# --------------------------------------------------------------------------
# compare_metrics: the agent-vs-human scoring itself
# --------------------------------------------------------------------------
from src.muzero.human_baseline import HumanLevelStats, compare_metrics  # noqa: E402

HUMAN = HumanLevelStats(
    level="Level1-3", n_reps=150, n_segments=298, n_completed=78, n_no_death=53,
    completion_rate=0.52, no_death_rate=0.353,
    length_mean=648.0, length_median=645.0, length_min=201.0, length_p10=313.0,
    return_mean=225.8, return_median=243.3,
)


def test_compare_metrics_empty_for_unplayed_level():
    assert compare_metrics("Level2-2", completed=True, n_steps=1, total_return=1.0,
                           stats=None, completion_bonus=200.0) == {}


def test_compare_metrics_faster_than_median_human():
    m = compare_metrics("Level1-3", completed=True, n_steps=476, total_return=259.25,
                        stats=HUMAN, completion_bonus=200.0)
    assert m["human/Level1-3_completion_rate_any_life"] == pytest.approx(0.52)
    assert m["human/Level1-3_completion_rate_first_life"] == pytest.approx(0.353)
    # compared against the FIRST-LIFE rate: agent episodes are single-life
    assert m["compare/Level1-3_completed_vs_human"] == pytest.approx(1.0 - 0.353)
    # human return is rebased from the converter's bonus (100) to this run's (200)
    assert m["human/Level1-3_return_median"] == pytest.approx(253.3)
    assert m["compare/Level1-3_return_ratio"] == pytest.approx(259.25 / 253.3)
    assert m["compare/Level1-3_length_ratio"] == pytest.approx(476 / 645)
    assert m["compare/Level1-3_beats_human_median"] == 1.0
    assert m["compare/Level1-3_beats_human_p10"] == 0.0  # 476 > p10 of 313


def test_compare_metrics_omits_length_when_agent_died():
    m = compare_metrics("Level1-3", completed=False, n_steps=297, total_return=141.0,
                        stats=HUMAN, completion_bonus=200.0)
    # time-to-flag is meaningless without a flag
    assert not any(k.startswith("compare/Level1-3_length") for k in m)
    assert "compare/Level1-3_beats_human_median" not in m
    # but progress-so-far vs the human return still is
    assert m["compare/Level1-3_return_ratio"] == pytest.approx(141.0 / 253.3)
    assert m["compare/Level1-3_completed_vs_human"] == pytest.approx(-0.353)


def test_compare_metrics_without_human_completions_has_no_length_reference():
    stats = HumanLevelStats(level="Level8-4", n_reps=10, completion_rate=0.0)
    m = compare_metrics("Level8-4", completed=True, n_steps=100, total_return=50.0,
                        stats=stats, completion_bonus=200.0)
    assert m["compare/Level8-4_completed_vs_human"] == 1.0
    assert not any("length" in k or "return" in k for k in m)


def test_compare_uses_first_life_rate_not_per_rep():
    """Regression: agent episodes end on the first death, so a human who
    finished on their third life is not a comparable success."""
    stats = HumanLevelStats(
        level="Level1-2", n_reps=136, completion_rate=0.081, no_death_rate=0.059,
        n_completed=11, length_median=1154.0, length_p10=800.0, return_median=297.5,
    )
    m = compare_metrics("Level1-2", completed=False, n_steps=200, total_return=10.0,
                        stats=stats, completion_bonus=100.0)
    assert m["compare/Level1-2_completed_vs_human"] == pytest.approx(-0.059)
    assert m["compare/Level1-2_completed_vs_human"] != pytest.approx(-0.081)
