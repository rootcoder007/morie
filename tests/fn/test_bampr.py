"""Tests for morie.fn.bampr -- posterior summary."""

from morie.fn import _array_core as np
from morie.fn.bampr import bampr, bayesian_am_posterior_summary


def test_alias():
    assert bampr is bayesian_am_posterior_summary


def test_smoke():
    chain = np.random.default_rng(42).standard_normal((200, 3))
    r = bayesian_am_posterior_summary(chain)
    assert r.name == "bayesian_am_posterior_summary"
    assert len(r.extra["means"]) == 3
    assert len(r.extra["sds"]) == 3
    assert len(r.extra["ci_lo"]) == 3


def test_1d_chain():
    chain = np.random.default_rng(42).standard_normal(100)
    r = bayesian_am_posterior_summary(chain)
    assert r.extra["n_params"] == 1


def test_posterior_summary_recomputed():
    import math

    import pytest

    chain = [[0.1, 2.0], [0.4, 1.5], [0.3, 2.5], [0.9, 1.0], [0.6, 3.0]]
    r = bayesian_am_posterior_summary(chain)

    def q(v, p):
        s = sorted(v)
        h = (len(s) - 1) * p
        lo = int(h)
        return s[lo] + (h - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])

    for j in range(2):
        col = [row[j] for row in chain]
        m = sum(col) / 5
        assert r.extra["means"][j] == pytest.approx(m, rel=1e-14)
        assert r.extra["sds"][j] == pytest.approx(math.sqrt(sum((v - m) ** 2 for v in col) / 4), rel=1e-13)
        assert r.extra["ci_lo"][j] == pytest.approx(q(col, 0.025), rel=1e-13)
        assert r.extra["ci_hi"][j] == pytest.approx(q(col, 0.975), rel=1e-13)
