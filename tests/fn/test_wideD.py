"""Tests for wideD.wide_and_deep."""

import math

import pytest

from morie.fn.wideD import wide_and_deep

N = 40


def _data():
    """A learnable binary problem: 3 wide columns, 2 dense columns."""
    xw, xd, y = [], [], []
    for i in range(N):
        a = 1.0 if i % 2 == 0 else 0.0
        b = 1.0 if (i // 2) % 2 == 0 else 0.0
        c = (i % 5) * 0.25
        xw.append([a, b, c])
        xd.append([math.sin(0.3 * i), math.cos(0.5 * i)])
        # label driven by the wide part, so the wide coefficients must move
        y.append(1.0 if (a + b + 0.5 * c) > 1.0 else 0.0)
    return xw, xd, y


def _predict(res, xw_row, xd_row, hidden_out):
    """Re-run eqs (2) and (3) from the returned parameters."""
    W, B = res["hidden_weights"], res["hidden_bias"]
    a = list(xd_row)
    for layer in range(len(W)):
        a = [max(0.0, B[layer][u] + sum(W[layer][u][k] * a[k]
                                        for k in range(len(a))))
             for u in range(len(W[layer]))]
    assert len(a) == hidden_out
    z = res["bias"]
    z += sum(w * v for w, v in zip(res["coef_wide"], xw_row))
    z += sum(w * v for w, v in zip(res["coef_deep"], a))
    return 1.0 / (1.0 + math.exp(-z))


def test_wideD_basic():
    """fitted reproduces eq. (3) evaluated at the returned parameters."""
    xw, xd, y = _data()
    res = wide_and_deep(xw, xd, y, hidden=(4,), epochs=60, lr=0.2, seed=3)

    assert res["n"] == N
    assert res["n_wide"] == 3
    assert res["n_deep"] == 2
    assert res["epochs"] == 60
    assert len(res["coef_wide"]) == 3
    assert len(res["coef_deep"]) == 4
    assert len(res["hidden_weights"]) == 1
    assert len(res["hidden_weights"][0]) == 4
    assert all(len(row) == 2 for row in res["hidden_weights"][0])
    assert len(res["hidden_bias"][0]) == 4

    fitted = res["fitted"]
    assert len(fitted) == N
    for i in range(N):
        assert 0.0 < fitted[i] < 1.0
        assert fitted[i] == pytest.approx(_predict(res, xw[i], xd[i], 4),
                                          abs=1e-12)

    # the two parts are one model on one logistic loss: the joint fit
    # separates the classes better than a coin
    ones = [f for f, t in zip(fitted, y) if t == 1.0]
    zeros = [f for f, t in zip(fitted, y) if t == 0.0]
    assert ones and zeros
    assert sum(ones) / len(ones) > sum(zeros) / len(zeros)

    # the wide part carries the signal, so its coefficients left zero
    assert max(abs(v) for v in res["coef_wide"]) > 1e-3


def test_wideD_first_epoch_loss_is_log_two():
    """All output weights start at zero, so eq. (3) starts at p = 1/2."""
    xw, xd, y = _data()
    one = wide_and_deep(xw, xd, y, hidden=(4,), epochs=1, lr=0.2, seed=3)
    assert one["loss"] == pytest.approx(math.log(2.0), abs=1e-12)

    # gradient descent on that loss: more epochs, lower reported loss
    more = wide_and_deep(xw, xd, y, hidden=(4,), epochs=60, lr=0.2, seed=3)
    assert more["loss"] < one["loss"]

    # and the loss really is the mean logistic loss of the fit it reports
    recomputed = -sum(
        t * math.log(p) + (1.0 - t) * math.log(1.0 - p)
        for t, p in zip(y, more["fitted"])) / N
    assert recomputed < one["loss"]
    assert recomputed == pytest.approx(more["loss"], rel=0.05)


def test_wideD_crosses_widen_the_wide_design():
    """eq. (1): each index pair appends a cross-product column."""
    xw, xd, y = _data()
    plain = wide_and_deep(xw, xd, y, hidden=(4,), epochs=20, lr=0.2, seed=3)
    crossed = wide_and_deep(xw, xd, y, hidden=(4,), epochs=20, lr=0.2, seed=3,
                            crosses=[(0, 1), (0, 2)])
    assert len(plain["coef_wide"]) == 3
    assert len(crossed["coef_wide"]) == 5
    assert crossed["n_wide"] == 5
    # the extra columns change the fit
    assert crossed["fitted"] != plain["fitted"]
    for i in range(N):
        row = xw[i] + [xw[i][0] * xw[i][1], xw[i][0] * xw[i][2]]
        assert crossed["fitted"][i] == pytest.approx(
            _predict(crossed, row, xd[i], 4), abs=1e-12)


def test_wideD_is_deterministic_and_l2_shrinks():
    xw, xd, y = _data()
    kw = dict(hidden=(4,), epochs=30, lr=0.2)
    a = wide_and_deep(xw, xd, y, seed=3, **kw)
    b = wide_and_deep(xw, xd, y, seed=3, **kw)
    assert a["fitted"] == b["fitted"]
    assert a["coef_wide"] == b["coef_wide"]

    other = wide_and_deep(xw, xd, y, seed=99, **kw)
    assert other["hidden_weights"] != a["hidden_weights"]

    ridged = wide_and_deep(xw, xd, y, seed=3, l2=5.0, **kw)
    assert (max(abs(v) for v in ridged["coef_wide"])
            < max(abs(v) for v in a["coef_wide"]))


def test_wideD_edge():
    """Two rows, one hidden unit, and every documented rejection."""
    res = wide_and_deep([[1.0], [0.0]], [[0.5], [-0.5]], [1.0, 0.0],
                        hidden=(1,), epochs=5, lr=0.1)
    assert res["n"] == 2 and res["n_wide"] == 1 and res["n_deep"] == 1
    assert len(res["fitted"]) == 2
    assert res["fitted"][0] == pytest.approx(
        _predict(res, [1.0], [0.5], 1), abs=1e-12)

    xw, xd, y = _data()
    with pytest.raises(ValueError):
        wide_and_deep(xw, xd, [v + 0.5 for v in y], epochs=2)   # not binary
    with pytest.raises(ValueError):
        wide_and_deep(xw, xd, y[:-1], epochs=2)                 # row mismatch
    with pytest.raises(ValueError):
        wide_and_deep(xw[:1], xd[:1], y[:1], epochs=2)          # n < 2
    with pytest.raises(ValueError):
        wide_and_deep(xw, xd, y, epochs=0)
    with pytest.raises(ValueError):
        wide_and_deep(xw, xd, y, hidden=(0,), epochs=2)
    with pytest.raises(ValueError):
        wide_and_deep(xw, xd, y, epochs=2, crosses=[(0, 9)])
