"""The dumped teacher corpus must load through the imitation pipeline unchanged.

The trap this guards: `select_human_files` globs ``sub-*.npz``, so the naming
originally planned for T8 (``spec-w1l1_...``) would have been silently invisible
to the loader — no error, just an empty corpus.
"""
import numpy as np
import pytest

from src.muzero.human_data import (
    load_human_buffer,
    load_trajectory_npz,
    select_human_files,
)

A = 12


def _dump(path, T=24, level="Level3-3", completed=True):
    """A file shaped exactly as dump_specialist_trajectories.py writes one."""
    pol = np.random.dirichlet(np.ones(A), size=T).astype(np.float32)  # NOT one-hot
    rq = np.random.uniform(10, 200, T).astype(np.float32)
    ret = rq + np.random.uniform(-5, 5, T).astype(np.float32)
    np.savez_compressed(
        path,
        obs_stacks=np.random.randint(0, 255, (T, 4, 8, 8), dtype=np.uint8),
        actions=pol.argmax(1).astype(np.int64),
        rewards=np.random.randn(T).astype(np.float32),
        policies=pol, root_values=rq, returns=ret,
        priorities=np.abs(ret - rq).astype(np.float32) + 1e-3,
        level=level, terminal=True, completed=completed,
        bk2="spec-level3-3@548111", teacher_run="spec-level3-3",
        teacher_checkpoint="outputs/runs/spec-level3-3/checkpoints/best.pt",
        teacher_training_step="548111", rollout_seed="9001",
        rollout_eps="0.25", rollout_temperature="0.25",
    )


@pytest.fixture
def corpus(tmp_path):
    for rep in range(3):
        _dump(tmp_path / f"sub-spec-w3l3_ses-000_task-mario_level-w3l3_rep-{rep:03d}_seg0.npz")
    _dump(tmp_path / "sub-spec-w1l1_ses-000_task-mario_level-w1l1_rep-000_seg0.npz",
          level="Level1-1")
    return tmp_path


def test_the_sub_prefix_is_what_makes_the_glob_find_them(tmp_path):
    """Regression for the BACKLOG naming bug: without `sub-`, zero files load."""
    _dump(tmp_path / "spec-w3l3_ses-000_task-mario_level-w3l3_rep-000_seg0.npz")
    with pytest.raises(FileNotFoundError):
        select_human_files(tmp_path)
    _dump(tmp_path / "sub-spec-w3l3_ses-000_task-mario_level-w3l3_rep-000_seg0.npz")
    assert len(select_human_files(tmp_path)) == 1


def test_level_filter_uses_the_tag(corpus):
    assert len(select_human_files(corpus, levels=["Level3-3"])) == 3
    assert len(select_human_files(corpus, levels=["Level1-1"])) == 1
    assert len(select_human_files(corpus, levels=["Level2-2"])) == 0


def test_teacher_is_its_own_selectable_subject(corpus):
    assert len(select_human_files(corpus, subjects=["sub-spec-w3l3"])) == 3
    with pytest.raises(ValueError, match="sub-04"):
        select_human_files(corpus, subjects=["sub-04"])


def test_trajectory_round_trips_with_extra_metadata_ignored(corpus):
    f = sorted(corpus.glob("sub-spec-w3l3*"))[0]
    traj, completed = load_trajectory_npz(f)
    assert completed is True and traj.level == "Level3-3" and traj.terminal is True
    assert traj.obs_stacks.dtype == np.uint8 and traj.policies.shape[1] == A


def test_policies_are_visit_distributions_not_one_hot(corpus):
    """The whole point of a specialist teacher over the human corpus."""
    traj, _ = load_trajectory_npz(sorted(corpus.glob("sub-spec-w3l3*"))[0])
    p = traj.policies
    assert np.allclose(p.sum(1), 1.0)
    assert (p.max(1) == 1.0).sum() == 0


def test_buffer_builds_and_samples(corpus):
    buf, ev = load_human_buffer(corpus, unroll_K=5, num_actions=A,
                                levels=["Level3-3"], holdout_fraction=0.34,
                                holdout_max_transitions=16)
    assert buf.size_transitions() > 0 and ev is not None and len(ev) > 0
    batch = buf.sample(2, 0)
    pol = np.asarray(batch["policies"]).reshape(-1, A)
    assert np.allclose(pol.sum(-1), 1.0)
    assert (pol.max(-1) == 1.0).sum() == 0   # still distillation targets after sampling
