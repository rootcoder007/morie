"""Tests for chrbnd.chernozhukov_rosen_bounds."""

from morie.fn import _array_core as np
from morie.fn.chrbnd import chernozhukov_rosen_bounds


def test_chrbnd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = chernozhukov_rosen_bounds(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_chrbnd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = chernozhukov_rosen_bounds(y)
    assert isinstance(result, dict)


def test_intersection_bound_recomputed():
    """CLR with independent cells: k_S(p) = qnorm(p^(1/|S|)), gamma_n =
    1 - 0.1/log n, contact set m_v <= min(m_u + k s_u) + 2 k s_v, bound
    min over the contact set of m_v + k_(contact)(1 - alpha) s_v."""
    import math

    import pytest

    from morie.fn._s03core import qnorm

    y = [3.0, 3.5, 2.8, 1.0, 1.6, 1.2, 1.4, 5.0, 4.1, 4.6]
    cell = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    means, ses = [], []
    for c in (0, 1, 2):
        v = [y[i] for i in range(10) if cell[i] == c]
        m = sum(v) / len(v)
        means.append(m)
        ses.append(math.sqrt(sum((t - m) ** 2 for t in v) / (len(v) - 1)) / math.sqrt(len(v)))
    g = 1 - 0.1 / math.log(10)
    kg = qnorm(g ** (1 / 3))
    thr = min(means[v] + kg * ses[v] for v in range(3))
    contact = [v for v in range(3) if means[v] <= thr + 2 * kg * ses[v]]
    ka = qnorm(0.95 ** (1 / len(contact)))
    r = chernozhukov_rosen_bounds(y, instrument=cell)
    assert r["contact_set"] == contact
    assert r["bound"] == pytest.approx(min(means[v] + ka * ses[v] for v in contact), rel=1e-12)
    kh = qnorm(0.5 ** (1 / len(contact)))
    assert r["hmu_estimate"] == pytest.approx(min(means[v] + kh * ses[v] for v in contact), rel=1e-12)
    assert r["naive_min"] == pytest.approx(min(means), rel=1e-14)
    # the precision correction moves the estimate UP from the naive minimum
    assert r["hmu_estimate"] >= r["naive_min"]
