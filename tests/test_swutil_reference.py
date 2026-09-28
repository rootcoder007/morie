"""swutil against spdep 1.3 (nb2listw styles, spweights.constants, card) and spatialreg (eigenw).

Reference values printed by R for the weighted matrix W below (unit 6 is an
island); tests/cross/test-morie_vs_spdep_weights.R repeats this live.
"""

import math

import pytest

from morie.fn.swutil import sinkhorn_weights, weights_style, weights_summary

W = [
    [0, 1.5, 0, 0.5, 0, 0],
    [1.5, 0, 2, 0, 0, 0],
    [0, 2, 0, 1, 0, 0],
    [0.5, 0, 1, 0, 0.7, 0],
    [0, 0, 0, 0.7, 0, 0],
    [0, 0, 0, 0, 0, 0],
]
STYLES = {
    "W": [
        0,
        0.75,
        0,
        0.25,
        0,
        0,
        0.42857142857142855,
        0,
        0.5714285714285714,
        0,
        0,
        0,
        0,
        0.66666666666666663,
        0,
        0.33333333333333331,
        0,
        0,
        0.22727272727272727,
        0,
        0.45454545454545453,
        0,
        0.31818181818181818,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    "B": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "C": [
        0,
        0.6578947368421052,
        0,
        0.21929824561403508,
        0,
        0,
        0.6578947368421052,
        0,
        0.8771929824561403,
        0,
        0,
        0,
        0,
        0.8771929824561403,
        0,
        0.43859649122807015,
        0,
        0,
        0.21929824561403508,
        0,
        0.43859649122807015,
        0,
        0.30701754385964908,
        0,
        0,
        0,
        0,
        0.30701754385964908,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    "U": [
        0,
        0.13157894736842105,
        0,
        0.043859649122807015,
        0,
        0,
        0.13157894736842105,
        0,
        0.17543859649122806,
        0,
        0,
        0,
        0,
        0.17543859649122806,
        0,
        0.08771929824561403,
        0,
        0,
        0.043859649122807015,
        0,
        0.08771929824561403,
        0,
        0.061403508771929814,
        0,
        0,
        0,
        0,
        0.061403508771929814,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    "minmax": [
        0,
        0.42857142857142855,
        0,
        0.14285714285714285,
        0,
        0,
        0.42857142857142855,
        0,
        0.5714285714285714,
        0,
        0,
        0,
        0,
        0.5714285714285714,
        0,
        0.2857142857142857,
        0,
        0,
        0.14285714285714285,
        0,
        0.2857142857142857,
        0,
        0.19999999999999998,
        0,
        0,
        0,
        0,
        0.19999999999999998,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    "S": [
        0,
        0.85282983723880523,
        0,
        0.28427661241293511,
        0,
        0,
        0.5393769484450619,
        0,
        0.71916926459341579,
        0,
        0,
        0,
        0,
        0.80405568147970508,
        0,
        0.40202784073985254,
        0,
        0,
        0.34075050780646698,
        0,
        0.68150101561293397,
        0,
        0.47705071092905382,
        0,
        0,
        0,
        0,
        0.89896158074176968,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
}
CONST_W = [5, 5.5080559772118214, 21.097882817038659]
EIG_W = [
    -0.99999999999999822,
    -0.47843087682141028,
    -1.5003759331839197e-16,
    0,
    0.47843087682141078,
    0.99999999999999822,
]
CARD = [2, 2, 2, 3, 1, 0]


@pytest.mark.parametrize("style", sorted(STYLES))
def test_styles_equal_spdep(style):
    got = [v for row in weights_style(W, style) for v in row]
    assert got == pytest.approx(STYLES[style], abs=1e-15)


def test_constants_cardinality_eigenvalues():
    s = weights_summary(weights_style(W, "W"))
    assert pytest.approx(CONST_W, abs=1e-13) == [s.S0, s.S1, s.S2]
    assert s.eigenvalues == pytest.approx(EIG_W, abs=1e-12)
    b = weights_summary(W)
    assert b.cardinality == CARD
    assert b.islands == [5]
    assert b.nonzero == 10 and b.symmetric and not b.row_stochastic
    assert b.trace_W2 == pytest.approx(sum(W[i][j] * W[j][i] for i in range(6) for j in range(6)), abs=1e-15)
    assert b.frobenius == pytest.approx(math.sqrt(sum(v * v for r in W for v in r)), abs=1e-15)
    assert b.lower_triangle[3] == [0.5, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert weights_summary(weights_style([r[:5] for r in W[:5]], "W")).row_stochastic


def test_asymmetry_and_diagonal_dominance():
    A = [[0.0, 1.0], [0.0, 0.0]]
    s = weights_summary(A)
    assert not s.symmetric
    assert s.asymmetry == pytest.approx(math.sqrt(0.5) / 1.0, abs=1e-15)
    assert weights_summary([[2.0, 1.0], [0.5, 1.0]], eigen=False).diag_dominant


def test_sinkhorn_is_doubly_stochastic():
    # units 1-4 form a 4-cycle, so the block has total support
    r = sinkhorn_weights([r[:4] for r in W[:4]])
    M = r.W
    assert max(abs(sum(row) - 1) for row in M) < 1e-12
    assert max(abs(sum(M[i][j] for i in range(4)) - 1) for j in range(4)) < 1e-12
    # a symmetric matrix scales to a symmetric doubly-stochastic one: D W D
    assert max(abs(M[i][j] - M[j][i]) for i in range(4) for j in range(4)) < 1e-10


def test_validation():
    with pytest.raises(ValueError):
        weights_style([[0, 1]], "W")
    with pytest.raises(ValueError):
        weights_style(W, "X")
    with pytest.raises(ValueError):
        sinkhorn_weights([[0, -1], [1, 0]])
