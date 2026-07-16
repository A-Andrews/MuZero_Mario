import numpy as np
import pytest

from src.muzero.buffer import Trajectory, TrajectoryBuffer
from src.muzero.human_data import (
    level_filename_tag,
    load_human_buffer,
    load_trajectory_npz,
    select_human_files,
)

A = 12


def _write_npz(path, T=10, level="Level1-1", terminal=True, completed=False):
    """Mimic scripts/convert_human_bk2.py output, incl. the extra metadata keys
    and 0-d string/bool scalars."""
    actions = np.random.randint(0, A, size=T).astype(np.int64)
    policies = np.zeros((T, A), dtype=np.float32)
    policies[np.arange(T), actions] = 1.0
    np.savez_compressed(
        path,
        obs_stacks=np.random.randint(0, 255, size=(T, 4, 8, 8), dtype=np.uint8),
        actions=actions,
        rewards=np.random.randn(T).astype(np.float32),
        policies=policies,
        root_values=np.zeros(T, dtype=np.float32),
        returns=np.zeros(T, dtype=np.float32),
        priorities=np.ones(T, dtype=np.float32),
        level=level,
        terminal=terminal,
        completed=completed,
        bk2=path.stem + ".bk2",
    )


@pytest.fixture
def corpus(tmp_path):
    """A tiny corpus: 2 subjects x 2 levels x 2 reps, one completed rep."""
    specs = []
    for sub in ("sub-01", "sub-02"):
        for w, l, level in ((1, 1, "Level1-1"), (2, 3, "Level2-3")):
            for rep in range(2):
                name = f"{sub}_ses-001_task-mario_level-w{w}l{l}_rep-{rep:03d}_seg0.npz"
                completed = sub == "sub-01" and level == "Level1-1" and rep == 0
                _write_npz(tmp_path / name, T=10, level=level, completed=completed)
                specs.append((name, level, completed))
    return tmp_path, specs


def test_load_trajectory_npz_filters_and_casts(corpus):
    data_dir, specs = corpus
    traj, completed = load_trajectory_npz(data_dir / specs[0][0])
    assert isinstance(traj, Trajectory)
    assert isinstance(traj.level, str) and traj.level == "Level1-1"
    assert isinstance(traj.terminal, bool)
    assert isinstance(completed, bool)
    assert traj.obs_stacks.dtype == np.uint8
    assert traj.policies.shape == (10, A)


def test_level_filename_tag():
    assert level_filename_tag("Level1-1") == "level-w1l1"
    assert level_filename_tag("Level8-3") == "level-w8l3"
    with pytest.raises(ValueError):
        level_filename_tag("w1l1")


def test_select_subject_filter(corpus):
    data_dir, _ = corpus
    files = select_human_files(data_dir, subjects=["sub-01"])
    assert len(files) == 4
    assert all(f.name.startswith("sub-01_") for f in files)


def test_select_unknown_subject_raises(corpus):
    data_dir, _ = corpus
    with pytest.raises(ValueError, match="sub-04"):
        select_human_files(data_dir, subjects=["sub-04"])


def test_select_level_filter(corpus):
    data_dir, _ = corpus
    files = select_human_files(data_dir, levels=["Level2-3"])
    assert len(files) == 4
    assert all("_level-w2l3_" in f.name for f in files)


def test_load_human_buffer_basic(corpus):
    data_dir, _ = corpus
    buf, eval_set = load_human_buffer(
        data_dir, unroll_K=3, num_actions=A, holdout_fraction=0.0
    )
    assert eval_set is None
    assert buf.size_transitions() == 8 * 10
    assert buf.size_trajectories() == 8
    # Pinned: capacity equals contents, sampling works.
    assert buf.capacity == buf.size_transitions()
    batch = buf.sample(batch_size=5, train_step=0)
    assert batch["obs"].shape == (5, 4, 8, 8)
    # All-ones priorities + alpha=1 -> uniform sampling, IS weights == 1.
    assert np.allclose(batch["is_weights"], 1.0)


def test_completed_only(corpus):
    data_dir, _ = corpus
    buf, _ = load_human_buffer(
        data_dir, unroll_K=3, num_actions=A, completed_only=True
    )
    assert buf.size_trajectories() == 1


def test_max_transitions_cap(corpus):
    data_dir, _ = corpus
    buf, _ = load_human_buffer(
        data_dir, unroll_K=3, num_actions=A, max_transitions=25
    )
    # Whole-trajectory granularity: stops adding once the cap is reached.
    assert 25 <= buf.size_transitions() <= 35


def test_holdout_disjoint_and_capped(corpus):
    data_dir, _ = corpus
    buf, eval_set = load_human_buffer(
        data_dir, unroll_K=3, num_actions=A,
        holdout_fraction=0.25, holdout_max_transitions=15,
    )
    assert eval_set is not None
    assert len(eval_set) == 15
    assert eval_set.obs.dtype == np.uint8
    assert eval_set.actions.dtype == np.int64
    # 2 of 8 files held out -> 6 files / 60 transitions trainable.
    assert buf.size_trajectories() == 6


def test_zero_after_filtering_raises(corpus):
    data_dir, _ = corpus
    with pytest.raises(ValueError, match="0 human transitions"):
        load_human_buffer(
            data_dir, unroll_K=3, num_actions=A,
            subjects=["sub-02"], completed_only=True,
        )


def test_bulk_add_rebuild_false_matches_per_add(corpus):
    data_dir, specs = corpus
    trajs = [load_trajectory_npz(data_dir / name)[0] for name, _, _ in specs]

    per_add = TrajectoryBuffer(capacity_transitions=10_000, unroll_K=3, num_actions=A)
    for t in trajs:
        per_add.add(t)

    trajs2 = [load_trajectory_npz(data_dir / name)[0] for name, _, _ in specs]
    bulk = TrajectoryBuffer(capacity_transitions=10_000, unroll_K=3, num_actions=A)
    for t in trajs2:
        bulk.add(t, rebuild=False)
    bulk._rebuild_flat()

    np.testing.assert_array_equal(per_add._flat_priorities, bulk._flat_priorities)
    np.testing.assert_array_equal(per_add._flat_traj_ids, bulk._flat_traj_ids)
    np.testing.assert_array_equal(per_add._flat_positions, bulk._flat_positions)
