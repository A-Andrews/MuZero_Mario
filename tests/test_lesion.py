"""The lesion contract: ablating one target must leave everything else alone.

Mirrors `tests/test_networks.py::TestAblationContract` in the Hanoi study. The
failure this guards against is silent: Mario's policy and value heads sit behind
a *shared* residual trunk, so a "policy lesion" written carelessly (resetting
`prediction` wholesale) also destroys value, and the experiment then reports a
policy deficit that is really a policy+value deficit.
"""
import torch

from src.muzero.lesion import (
    CONDITIONS,
    LESION_TARGETS,
    apply_lesion,
    lesioned_parameter_names,
    resolve_targets,
)
from src.muzero.networks import MuZeroNet


def _net(seed=0):
    torch.manual_seed(seed)
    # Small but structurally identical to the trained runs (192ch/10 blocks).
    return MuZeroNet(hidden_channels=16, hidden_spatial=6, num_actions=12,
                     value_support=(-25.0, 25.0, 21), reward_support=(-8.0, 8.0, 11),
                     rep_blocks=(1, 1, 1, 1), dyn_blocks=2, pred_blocks=1).eval()


def _snapshot(net):
    return {n: p.detach().clone() for n, p in
            list(net.named_parameters()) + list(net.named_buffers())}


class TestLesionContract:
    def test_every_target_resolves(self):
        """LESION_TARGETS must not go stale against the real architecture."""
        net = _net()
        for target in LESION_TARGETS:
            assert resolve_targets(net, [target]), target

    def test_lesion_changes_its_own_parameters(self):
        net = _net()
        for target in LESION_TARGETS:
            fresh = _net()
            before = _snapshot(fresh)
            apply_lesion(fresh, [target], seed=1)
            touched = lesioned_parameter_names(fresh, [target])
            after = _snapshot(fresh)
            changed = {n for n in before if not torch.equal(before[n], after[n])}
            # Weights must actually move. Buffers (BN running stats) may already
            # equal their reset values in a freshly built net, so only require
            # that *something* inside the target changed.
            assert changed, f"lesioning {target!r} changed nothing"
            assert changed <= touched, (
                f"lesioning {target!r} changed parameters outside itself: "
                f"{sorted(changed - touched)}")

    def test_everything_else_is_bit_identical(self):
        """The core contract, checked for every single-target lesion."""
        for target in LESION_TARGETS:
            net = _net()
            before = _snapshot(net)
            apply_lesion(net, [target], seed=2)
            after = _snapshot(net)
            protected = set(before) - lesioned_parameter_names(net, [target])
            for name in protected:
                assert torch.equal(before[name], after[name]), (
                    f"lesioning {target!r} altered {name!r}, which it must not touch")

    def test_policy_lesion_spares_value_and_the_shared_trunk(self):
        """The specific mistake this module exists to prevent."""
        net = _net()
        before = _snapshot(net)
        apply_lesion(net, ["policy"], seed=3)
        after = _snapshot(net)
        for name in before:
            if name.startswith("prediction.value") or name.startswith("prediction.blocks"):
                assert torch.equal(before[name], after[name]), (
                    f"policy lesion damaged {name!r} — value and the shared trunk "
                    f"must survive it")

    def test_reward_lesion_spares_the_transition_path(self):
        """A reward lesion must not degrade the forward model MCTS rolls out."""
        net = _net()
        before = _snapshot(net)
        apply_lesion(net, ["reward"], seed=4)
        after = _snapshot(net)
        for name in before:
            if name.startswith(("dynamics.conv_in", "dynamics.bn_in", "dynamics.blocks")):
                assert torch.equal(before[name], after[name]), (
                    f"reward lesion damaged {name!r} in the transition path")

    def test_intact_condition_is_a_no_op(self):
        net = _net()
        before = _snapshot(net)
        rec = apply_lesion(net, CONDITIONS["intact"], seed=5)
        after = _snapshot(net)
        assert rec["params_reset"] == 0
        for name in before:
            assert torch.equal(before[name], after[name])

    def test_combination_equals_union_of_its_parts(self):
        """policy+value must touch exactly what policy and value touch separately."""
        net = _net()
        combo = lesioned_parameter_names(net, ["policy", "value"])
        parts = (lesioned_parameter_names(net, ["policy"])
                 | lesioned_parameter_names(net, ["value"]))
        assert combo == parts

    def test_seed_is_reproducible_and_varies(self):
        a, b, c = _net(), _net(), _net()
        apply_lesion(a, ["value"], seed=7)
        apply_lesion(b, ["value"], seed=7)
        apply_lesion(c, ["value"], seed=8)
        sa, sb, sc = _snapshot(a), _snapshot(b), _snapshot(c)
        assert all(torch.equal(sa[n], sb[n]) for n in sa), "same seed must reproduce"
        assert any(not torch.equal(sa[n], sc[n]) for n in sa), "different seed must differ"

    def test_lesion_does_not_disturb_global_rng(self):
        """Re-init must not shift the RNG stream the rollout seeds come from."""
        net = _net()
        torch.manual_seed(1234)
        expected = torch.randn(4)
        torch.manual_seed(1234)
        apply_lesion(net, ["encoder"], seed=99)
        assert torch.equal(torch.randn(4), expected)


class TestLesionChangesBehaviour:
    def test_forward_pass_still_runs_and_output_moves(self):
        """A lesioned net must stay usable, and must actually behave differently.

        The input must not be constant: the representation net ends in a
        per-sample min-max normalisation, so a spatially uniform input (zeros)
        collapses to the same output whatever the encoder weights are, and an
        encoder lesion would look like a no-op.
        """
        torch.manual_seed(0)
        obs = torch.rand(2, 4, 96, 96)
        for target in LESION_TARGETS:
            net = _net()
            with torch.no_grad():
                h0, p0, v0 = net.initial_inference(obs)
            apply_lesion(net, [target], seed=11)
            with torch.no_grad():
                h1, p1, v1 = net.initial_inference(obs)
            assert h1.shape == h0.shape and p1.shape == p0.shape
            if target in ("encoder", "pred_trunk", "policy", "value"):
                moved = (not torch.equal(h0, h1)) or (not torch.equal(p0, p1)) \
                        or (not torch.equal(v0, v1))
                assert moved, f"{target!r} lesion left initial_inference unchanged"

    def test_transition_lesion_changes_recurrent_not_initial(self):
        """Damaging the forward model must not touch the root evaluation."""
        torch.manual_seed(0)
        obs = torch.rand(2, 4, 96, 96)
        net = _net()
        with torch.no_grad():
            h0, p0, v0 = net.initial_inference(obs)
            a = torch.zeros(2, dtype=torch.long)
            hn0, r0, _, _ = net.recurrent_inference(h0, a)
        apply_lesion(net, ["transition"], seed=13)
        with torch.no_grad():
            h1, p1, v1 = net.initial_inference(obs)
            hn1, r1, _, _ = net.recurrent_inference(h1, a)
        assert torch.equal(h0, h1) and torch.equal(p0, p1) and torch.equal(v0, v1), \
            "transition lesion must leave the root (encoder + prediction) intact"
        assert not torch.equal(hn0, hn1), "transition lesion must change imagination"
