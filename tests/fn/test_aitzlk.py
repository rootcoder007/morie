"""Tests for aitzlk.compositional_zero_lrda."""

from morie.fn.aitzlk import compositional_zero_lrda, lrda

# Six three-part compositions; the third part is never censored because
# its detection limit is zero, so it can serve as the alr denominator.
X = [[0.20, 1.00, 2.00],
     [1.50, 0.30, 1.20],
     [0.90, 1.10, 1.00],
     [0.10, 0.20, 1.70],
     [2.00, 2.00, 1.00],
     [0.35, 1.40, 0.90]]
DL = [0.4, 0.4, 0.0]
N_ITER = 5
DRAW = [[0.1 * (i - 2) + 0.05 * t for i in range(len(X))]
        for t in range(N_ITER)]

CENSORED = [(i, j) for i in range(len(X)) for j in range(3)
            if X[i][j] < DL[j]]


def test_aitzlk_basic():
    """lrDA imputes only the censored parts and keeps each row total."""
    assert lrda is compositional_zero_lrda
    res = compositional_zero_lrda(X, DL, DRAW, n_iter=N_ITER)
    assert res["n"] == len(X)
    assert res["n_parts"] == 3
    assert res["n_iter"] == N_ITER
    assert res["n_censored"] == len(CENSORED) == 5
    W = res["X"]
    assert len(W) == len(X) and all(len(r) == 3 for r in W)
    # alr_inv rescales each row back to the total it came in with
    for i, row in enumerate(W):
        assert abs(sum(row) - sum(X[i])) < 1e-12
        assert all(v > 0.0 for v in row)
    # rows with nothing below the limit are returned untouched
    assert W[2] == X[2]
    assert W[4] == X[4]
    # every imputed value respects the limit it was truncated at
    for i, j in CENSORED:
        assert W[i][j] <= DL[j] + 1e-12
        assert W[i][j] > 0.0
    # the caller supplies the variates, so a run is reproducible
    assert compositional_zero_lrda(X, DL, DRAW, n_iter=N_ITER)["X"] == W


def test_aitzlk_edge():
    """The draws are used, and bad arguments are rejected."""
    lo = compositional_zero_lrda(
        X, DL, [[-1.5] * len(X) for _ in range(N_ITER)], n_iter=N_ITER)["X"]
    hi = compositional_zero_lrda(
        X, DL, [[1.5] * len(X) for _ in range(N_ITER)], n_iter=N_ITER)["X"]
    # a lower normal variate imputes strictly smaller censored parts
    for i, j in CENSORED:
        assert lo[i][j] < hi[i][j]
    for i in (2, 4):
        assert lo[i] == X[i] and hi[i] == X[i]
    # n_iter=0 leaves the paper initial fill of 65% of the limit
    zero = compositional_zero_lrda(X, DL, DRAW, n_iter=0)["X"]
    for i, j in CENSORED:
        assert abs(zero[i][j] - 0.65 * DL[j]) < 1e-12
    for i, j in [(i, j) for i in range(len(X)) for j in range(3)
                 if (i, j) not in CENSORED]:
        assert zero[i][j] == X[i][j]
    try:
        compositional_zero_lrda(X, DL, [])
        raise AssertionError("expected ValueError for no variates")
    except ValueError:
        pass
    try:
        compositional_zero_lrda(X, [0.4, 0.4], DRAW)
        raise AssertionError("expected ValueError for short dl")
    except ValueError:
        pass
