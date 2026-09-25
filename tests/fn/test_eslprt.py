"""Tests for eslprt.esl_partial_dependence."""

import pytest

from morie.fn.eslprt import esl_partial_dependence


def _additive(Z):
    """f(z) = 3*z0 + z1**2, additive in its two arguments."""
    return [3.0 * float(Z[i][0]) + float(Z[i][1]) ** 2 for i in range(len(Z))]


def _grid_x():
    """Forty rows on a lattice, so the two columns are uncorrelated."""
    return [[-2.0 + 4.0 * (i % 8) / 7.0, -1.0 + 2.0 * (i // 8) / 4.0]
            for i in range(40)]


def test_eslprt_basic():
    """For an additive model the curve reproduces the component exactly."""
    X = _grid_x()
    result = esl_partial_dependence(_additive, X, S=0, n_grid=5)

    G = result["grid"]
    curve = result["pd"]
    assert G.shape == (5, 1)
    assert len(curve) == 5
    assert result["n"] == 40
    assert [int(s) for s in result["S"]] == [0]

    # pd(g) = 3g + mean(x1^2), so the curve is a straight line of slope 3.
    mean_sq = sum(row[1] ** 2 for row in X) / len(X)
    for t in range(5):
        assert abs(float(curve[t]) - (3.0 * float(G[t][0]) + mean_sq)) < 1e-9

    slope = (float(curve[4]) - float(curve[0])) / (float(G[4][0]) - float(G[0][0]))
    assert abs(slope - 3.0) < 1e-9

    # Centring makes the curve mean-zero.
    centered = result["centered"]
    assert abs(sum(float(v) for v in centered) / len(centered)) < 1e-9
    mean_pd = sum(float(v) for v in curve) / len(curve)
    for t in range(5):
        assert abs(float(centered[t]) - (float(curve[t]) - mean_pd)) < 1e-9

    # The two columns are on a lattice, so nothing is extrapolated.
    assert not any(bool(w) for w in result["extrapolation_warning"])


def test_eslprt_edge():
    """An explicit grid, correlated inputs, and the documented checks."""
    X = _grid_x()
    # An explicit grid is used as given.
    r = esl_partial_dependence(_additive, X, S=1, grid=[-1.0, 0.0, 1.0])
    mean_lin = sum(row[0] for row in X) / len(X)
    for t, g in enumerate((-1.0, 0.0, 1.0)):
        assert abs(float(r["pd"][t]) - (3.0 * mean_lin + g * g)) < 1e-9

    # Perfectly collinear inputs: moving x0 alone lands on combinations that
    # never occur, and those grid points are flagged.
    Xc = [[-2.0 + 4.0 * i / 39.0, -2.0 + 4.0 * i / 39.0] for i in range(40)]
    rc = esl_partial_dependence(_additive, Xc, S=0, n_grid=5)
    assert any(bool(w) for w in rc["extrapolation_warning"])

    with pytest.raises(ValueError, match="outside 0.."):
        esl_partial_dependence(_additive, X, S=2)
    with pytest.raises(ValueError, match="must not repeat"):
        esl_partial_dependence(_additive, X, S=[0, 0])
