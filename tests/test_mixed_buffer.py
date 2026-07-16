import numpy as np

from src.muzero.buffer import MixedBuffer, Trajectory, TrajectoryBuffer

A = 4


def _make_traj(T=10, priority=1.0):
    return Trajectory(
        obs_stacks=np.zeros((T, 4, 8, 8), dtype=np.uint8),
        actions=np.zeros(T, dtype=np.int64),
        rewards=np.zeros(T, dtype=np.float32),
        policies=np.ones((T, A), dtype=np.float32) / A,
        root_values=np.zeros(T, dtype=np.float32),
        returns=np.zeros(T, dtype=np.float32),
        priorities=np.full(T, priority, dtype=np.float32),
    )


def _make_buffers(reanalyze=False, main_trajs=3, human_trajs=3):
    kw = dict(unroll_K=3, num_actions=A)
    if reanalyze:
        kw.update(reanalyze=True, n_step=5, discount=0.99)
    main = TrajectoryBuffer(capacity_transitions=500, **kw)
    human = TrajectoryBuffer(capacity_transitions=500, **kw)
    for _ in range(main_trajs):
        main.add(_make_traj())
    for _ in range(human_trajs):
        human.add(_make_traj())
    return main, human


def test_mix_ratio_exact_counts():
    main, human = _make_buffers()
    mixed = MixedBuffer(main, human, mix_ratio=0.25)
    batch = mixed.sample(batch_size=8, train_step=0)
    sources = [loc[0] for loc in batch["sample_locations"]]
    assert sources.count("human") == 2  # ceil(8 * 0.25)
    assert sources.count("main") == 6


def test_all_keys_concatenated_with_full_batch_dim():
    main, human = _make_buffers(reanalyze=True)
    mixed = MixedBuffer(main, human, mix_ratio=0.5)
    B = 6
    batch = mixed.sample(batch_size=B, train_step=0)
    assert batch["obs"].shape == (B, 4, 8, 8)
    assert batch["actions"].shape == (B, 3)
    assert batch["rewards"].shape == (B, 3)
    assert batch["policies"].shape == (B, 4, A)
    assert batch["returns"].shape == (B, 4)
    assert batch["next_obs"].shape == (B, 3, 4, 8, 8)
    assert batch["next_obs"].dtype == np.uint8
    assert batch["next_obs_mask"].shape == (B, 3)
    assert batch["is_weights"].shape == (B,)
    assert batch["value_obs"].shape == (B, 4, 4, 8, 8)
    assert batch["value_obs_factor"].shape == (B, 4)
    assert batch["reward_window"].shape == (B, 4)
    assert len(batch["sample_locations"]) == B
    assert np.all(batch["is_weights"] <= 1.0)


def test_ratio_edges():
    main, human = _make_buffers()
    mixed = MixedBuffer(main, human, mix_ratio=0.0)
    sources = {loc[0] for loc in mixed.sample(5, 0)["sample_locations"]}
    assert sources == {"main"}

    mixed.set_mix_ratio(1.0)
    sources = {loc[0] for loc in mixed.sample(5, 0)["sample_locations"]}
    assert sources == {"human"}


def test_empty_main_falls_back_to_human():
    main, human = _make_buffers(main_trajs=0)
    # Pretrain regime: main buffer empty, any ratio must yield human-only batches.
    mixed = MixedBuffer(main, human, mix_ratio=0.25)
    batch = mixed.sample(batch_size=4, train_step=0)
    assert {loc[0] for loc in batch["sample_locations"]} == {"human"}
    assert batch["obs"].shape[0] == 4


def test_empty_human_falls_back_to_main():
    main, human = _make_buffers(human_trajs=0)
    mixed = MixedBuffer(main, human, mix_ratio=0.25)
    batch = mixed.sample(batch_size=4, train_step=0)
    assert {loc[0] for loc in batch["sample_locations"]} == {"main"}


def test_update_priorities_routes_to_correct_buffer():
    main, human = _make_buffers()
    mixed = MixedBuffer(main, human, mix_ratio=0.5)
    batch = mixed.sample(batch_size=8, train_step=0)

    main_before = main._flat_priorities.copy()
    human_before = human._flat_priorities.copy()
    mixed.update_priorities(
        batch["sample_locations"], np.full(8, 7.5, dtype=np.float32)
    )
    # Both sources were sampled, so both flat caches must have changed...
    assert not np.array_equal(main._flat_priorities, main_before)
    assert not np.array_equal(human._flat_priorities, human_before)
    # ...and every changed entry carries the new value.
    for buf, before in ((main, main_before), (human, human_before)):
        changed = buf._flat_priorities != before
        assert np.all(buf._flat_priorities[changed] == 7.5)

    # Routing respects tags: human-only updates leave main untouched.
    main_before = main._flat_priorities.copy()
    human_locs = [loc for loc in batch["sample_locations"] if loc[0] == "human"]
    mixed.update_priorities(human_locs, np.full(len(human_locs), 0.5, dtype=np.float32))
    np.testing.assert_array_equal(main._flat_priorities, main_before)


def test_add_and_sizes_are_main_only():
    main, human = _make_buffers(main_trajs=0)
    mixed = MixedBuffer(main, human, mix_ratio=0.25)
    assert mixed.size_transitions() == 0
    assert mixed.human_transitions() == 30
    mixed.add(_make_traj(T=10))
    assert mixed.size_transitions() == 10
    assert mixed.size_trajectories() == 1
    assert human.size_trajectories() == 3  # untouched
