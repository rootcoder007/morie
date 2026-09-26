"""Tests for morie.fn.n_logit -- minimum sample size for logistic regression."""

import pytest

from morie.fn.n_logit import sample_size_logistic


class TestSampleSizeLogistic:
    def test_returns_positive_int(self):
        """Should return a positive integer."""
        n = sample_size_logistic(p0=0.1, p1=0.2)
        assert isinstance(n, int)
        assert n > 0

    def test_smaller_effect_needs_more(self):
        """Smaller effect (closer p0, p1) needs larger n."""
        n_large_effect = sample_size_logistic(p0=0.1, p1=0.3)
        n_small_effect = sample_size_logistic(p0=0.1, p1=0.15)
        assert n_small_effect > n_large_effect

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError, match="p0"):
            sample_size_logistic(p0=0.0, p1=0.2)


# Reference: R powerMediation::SSizeLogisticBin(p1, p2, B, alpha, power) and
# SSizeLogisticCon(p1, OR, alpha, power), which implement Hsieh, Bloch &
# Larsen (1998) eqs. 1 and 2.
@pytest.mark.parametrize(
    "args, kw, ref",
    [
        ((0.2, 0.35), {}, 276),
        ((0.1, 0.18, 0.05, 0.9), {"B": 0.3}, 915),
        ((0.4, 0.3, 0.01, 0.8), {"B": 0.6}, 1097),
        ((0.2,), {"covariate": "continuous", "odds_ratio": 1.5}, 299),
        ((0.35,), {"power": 0.9, "covariate": "continuous", "odds_ratio": 0.7}, 364),
    ],
)
def test_matches_hsieh_1998_reference(args, kw, ref):
    assert sample_size_logistic(*args, **kw) == ref


def test_binary_formula_recomputed():
    import math

    from morie.fn import _stats_core as st

    p0, p1, B = 0.15, 0.25, 0.4
    za, zb = st.norm.ppf(0.975), st.norm.ppf(0.8)
    pbar = (1 - B) * p0 + B * p1
    n = (za * math.sqrt(pbar * (1 - pbar) / B) + zb * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1) * (1 - B) / B)) ** 2 / (
        (p0 - p1) ** 2 * (1 - B)
    )
    assert sample_size_logistic(p0, p1, B=B) == math.ceil(n)


def test_odds_ratio_gives_the_same_p1():
    # OR = [p1/(1-p1)] / [p0/(1-p0)]
    p0, p1 = 0.2, 0.35
    orr = (p1 / (1 - p1)) / (p0 / (1 - p0))
    assert sample_size_logistic(p0, odds_ratio=orr) == sample_size_logistic(p0, p1)
