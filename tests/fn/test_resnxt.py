"""Tests for resnxt.resnext_block (Xie et al. 2017, Sec. 3 forms (a)-(c))."""

import pytest

from morie.fn.resnxt import block_equivalence, block_parameters, match_complexity, resnext_block


X = [0.5, -1.0, 2.0]
WIN = [[[0.2, -0.1, 0.4], [0.3, 0.5, -0.2]], [[-0.6, 0.1, 0.2], [0.4, 0.4, 0.1]]]
WMID = [[[1.0, -0.5], [0.2, 0.3]], [[0.7, 0.1], [-0.2, 0.9]]]
WOUT = [[[0.1, 0.2], [-0.3, 0.4], [0.5, 0.0]], [[0.2, -0.1], [0.3, 0.3], [-0.4, 0.6]]]


def _path(i):
    h = [max(0.0, sum(WIN[i][o][j] * X[j] for j in range(3))) for o in range(2)]
    h = [max(0.0, sum(WMID[i][o][j] * h[j] for j in range(2))) for o in range(2)]
    return [sum(WOUT[i][o][j] * h[j] for j in range(2)) for o in range(3)]


def test_resnxt_basic():
    """y = x + sum_i T_i(x) with T_i = W_out relu(W_mid relu(W_in x)); the
    concatenate-then-project form (c) computes the same function; one
    bottleneck block has C (W d + 9 d^2 + d W) parameters and
    match_complexity solves 9 C d^2 + 2 C W d = budget for d."""
    y = resnext_block(X, WIN, WMID, WOUT)
    want = [X[j] + _path(0)[j] + _path(1)[j] for j in range(3)]
    assert y == pytest.approx(want, rel=1e-15)
    eq = block_equivalence(X, WIN, WMID, WOUT)
    assert eq["equivalent"] and eq["max_deviation"] < 1e-15
    assert block_parameters(256, 32, 4)["parameters"] == 32 * (256 * 4 + 9 * 16 + 4 * 256)
    m = match_complexity(256, 32, 70000)
    d = m["bottleneck"]
    assert 9 * 32 * d * d + 2 * 32 * 256 * d == pytest.approx(70000, rel=1e-12)


def test_resnxt_edge():
    """A cardinality of zero is rejected."""
    with pytest.raises(ValueError):
        block_parameters(256, 0, 4)
