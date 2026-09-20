import copy

import pytest

from scripts.eval_controller_diagnostic import select_plan
from tests.test_controller_diagnostic import manifest


def test_coverage_requires_both_arms_all_thirty_trials_and_control():
    data = manifest()
    data["evaluation_profile"] = "coverage_extension"
    data["conditions"] = [c for c in data["conditions"] if c["name"] in ("greedy", "sampled")]
    plan = select_plan(data)
    assert len(plan) == 60
    assert [c["name"] for c, _ in plan[:2]] == ["greedy", "sampled"]
    assert len({t["env_seed"] for _, t in plan}) == 30
    bad = copy.deepcopy(data)
    bad["conditions"] = bad["conditions"][:1]
    with pytest.raises(ValueError, match="declared controller"):
        select_plan(bad)
    bad = copy.deepcopy(data)
    del bad["levels"]["Level6-1"]
    with pytest.raises(ValueError, match="control"):
        select_plan(bad)
    with pytest.raises(ValueError, match="smoke"):
        select_plan(data, episodes=10)


def test_original_protocol_does_not_silently_accept_reduced_arms():
    data = manifest()
    data["conditions"] = data["conditions"][:2]
    with pytest.raises(ValueError, match="declared controller"):
        select_plan(data)
