import numpy as np
import pytest

from scripts.plot_policy_search_distributions import probability_counts, summarize


def test_probability_mass_is_not_argmax_frequency():
    probabilities=np.zeros((2,12))
    probabilities[:,1]=.51
    probabilities[:,2]=.49
    mass=probability_counts(probabilities)
    assert mass[1]==pytest.approx(1.02)
    assert mass[2]==pytest.approx(.98)
    assert np.bincount(probabilities.argmax(1),minlength=12)[1]==2


def test_episode_weighting_does_not_overweight_long_stalls():
    counts=np.zeros((2,12))
    counts[0,1]=1000
    counts[1,2]=10
    result=summarize(counts,[1000,10],[0,1])
    assert result['mean'][1:3]==[.5,.5]
    assert result['n_decisions']==1010
    assert result['pooled_decision_distribution'][1]>.99


def test_invalid_probabilities_are_rejected():
    for values in [np.zeros((2,12)), np.full((2,12),np.nan), np.full((2,11),1/11),np.full((0,12),1/12)]:
        with pytest.raises(ValueError):
            probability_counts(values)
    values=np.full((2,12),1/12)
    values[0,0]=-.1
    with pytest.raises(ValueError):
        probability_counts(values)
