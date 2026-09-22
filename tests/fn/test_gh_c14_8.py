"""Tests for gh_c14_8.ghosal_gibbs_proc."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_c14_8 import ghosal_gibbs_proc


def _pochhammer_log(x, m):
    """log of ascending factorial (x)_m = x*(x+1)*...*(x+m-1)."""
    if m <= 0:
        return 0.0
    s = 0.0
    for l in range(m):
        s += math.log(x + l)
    return s


def _expected_estimate(block_sizes, V_n_k, discount):
    ns = [int(v) for v in block_sizes]
    lp = math.log(float(V_n_k))
    for nj in ns:
        lp += _pochhammer_log(1.0 - discount, nj - 1)
    return math.exp(lp)


def test_gh_c14_8_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_gibbs_proc(x)
    assert "estimate" in result
    assert "log_prob" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    expected = _expected_estimate([1.0, 2.0, 3.0, 4.0, 5.0], 1.0, 0.5)
    assert np.isclose(result["estimate"], expected, rtol=1e-12)
    assert np.isclose(math.log(result["estimate"]), result["log_prob"], rtol=1e-12)


def test_gh_c14_8_edge():
    """Test edge case: single block."""
    block_sizes = np.array([42.0])
    result = ghosal_gibbs_proc(block_sizes, V_n_k=2.5, discount=0.3)
    assert "estimate" in result
    expected = _expected_estimate([42.0], 2.5, 0.3)
    assert np.isclose(result["estimate"], expected, rtol=1e-12)
    assert np.isclose(result["log_prob"], math.log(expected), rtol=1e-12)


def test_gh_c14_8_unit_blocks():
    """When every n_j == 1, every Pochhammer term is the empty product = 1."""
    x = np.array([1.0, 1.0, 1.0, 1.0])
    result = ghosal_gibbs_proc(x, V_n_k=7.0, discount=0.85)
    assert np.isclose(result["estimate"], 7.0, rtol=1e-12)
    assert np.isclose(result["log_prob"], math.log(7.0), rtol=1e-12)
