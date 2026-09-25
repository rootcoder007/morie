"""Tests for mehtad.mehrotras_predictor (standard-form LP, Mehrotra 1992)."""

import itertools

import pytest

from morie.fn.mehtad import mehrotras_predictor

# max x1 + 2 x2 s.t. x1 + x2 <= 4, x1 - x2 <= 2, x >= 0, in standard form
A = [[1.0, 1.0, 1.0, 0.0], [1.0, -1.0, 0.0, 1.0]]
B = [4.0, 2.0]
C = [-1.0, -2.0, 0.0, 0.0]


def _vertex_optimum():
    """Enumerate the basic feasible solutions (2 of 4 columns)."""
    best = None
    for i, j in itertools.combinations(range(4), 2):
        a, b, c, d = A[0][i], A[0][j], A[1][i], A[1][j]
        det = a * d - b * c
        if det == 0:
            continue
        xi, xj = (B[0] * d - b * B[1]) / det, (a * B[1] - B[0] * c) / det
        if xi < 0 or xj < 0:
            continue
        x = [0.0] * 4
        x[i], x[j] = xi, xj
        val = sum(cc * v for cc, v in zip(C, x))
        if best is None or val < best[0]:
            best = (val, x)
    return best


def test_mehtad_basic():
    """The interior-point solution converges on the optimal vertex found
    by enumerating every basis; primal and dual objectives meet. It stops
    once mu = x.s / n < 1e-9, so x.s < 4e-9; the optimal reduced costs of
    the nonbasic x1 and x3 are 1 and 2 (y = (-2, 0)), bounding each by
    4e-9, the basic ones move by the same amount through Ax = b, and the
    objective gap is x.s: 1e-8 covers all three."""
    val, x = _vertex_optimum()
    assert (val, x) == (-8.0, [0.0, 4.0, 0.0, 6.0])
    result = mehrotras_predictor(A, B, C)
    assert isinstance(result, dict)
    assert result["converged"]
    assert [float(v) for v in result["x"]] == pytest.approx(x, abs=1e-8)
    assert result["objective"] == pytest.approx(val, abs=1e-8)
    by = sum(b * y for b, y in zip(B, result["y"]))
    assert by == pytest.approx(val, abs=1e-8)


def test_mehtad_edge():
    """The predictor-only run reaches the same optimum; mismatched
    shapes are refused."""
    r = mehrotras_predictor(A, B, C, corrector=False)
    assert r["objective"] == pytest.approx(-8.0, abs=1e-8)
    with pytest.raises(ValueError):
        mehrotras_predictor(A, B + [1.0], C)


