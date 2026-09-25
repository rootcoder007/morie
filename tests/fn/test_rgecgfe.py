"""Tests for bsaclass.rangayyan_fetal_ecg_single (sec. 9.11, eqs. 9.88-9.91)."""

import math

import pytest

from morie.fn.bsaclass import rangayyan_fetal_ecg_single


FS = 250.0


def _g(t, c, w):
    return math.exp(-((t - c) ** 2) / (2 * w * w))


X = [_g((i % 200) / FS, 0.1, 0.01) + 0.3 * _g((i % 107) / FS, 0.2, 0.006) for i in range(2500)]


def _count(row, tau):
    m = max(abs(t) for t in row)
    nr = [abs(t) / m for t in row]
    return sum(1 for i in range(1, len(nr) - 1) if nr[i] > tau and nr[i] >= nr[i - 1] and nr[i] > nr[i + 1])


def test_rgecgfe_basic():
    """W and H are nonnegative; each normalised activation row's peaks
    above T = tau (eq. 9.91 with A_max = 1 after normalisation) are
    counted, and with the counts sorted ascending the smallest is the
    maternal row and the next the fetal row; the multiplicative updates
    (9.49)-(9.50) never increase the Frobenius error (Lee and Seung 2001);
    both reconstructions have the input length."""
    r = rangayyan_fetal_ecg_single(X, FS)
    assert all(v >= 0.0 for row in r["W"] for v in row)
    assert all(v >= 0.0 for row in r["H"] for v in row)
    counts = sorted((_count(h, 0.45), a) for a, h in enumerate(r["H"]))
    assert (r["maternalrow"], r["fetalrow"]) == (counts[0][1], counts[1][1])
    assert r["peaks"]["per_row"] == {a: c for c, a in counts}
    assert r["peaks"]["maternal_row_count_at_taum"] == _count(r["H"][r["maternalrow"]], 0.6)
    assert len(r["fetal"]) == len(r["maternal"]) == len(X)
    assert rangayyan_fetal_ecg_single(X, FS, maxiter=300)["error"] <= r["error"]


def test_rgecgfe_edge():
    """Thresholds outside (0, 1] and a negative sparsity raise."""
    with pytest.raises(ValueError):
        rangayyan_fetal_ecg_single(X, FS, taum=0.0)
    with pytest.raises(ValueError):
        rangayyan_fetal_ecg_single(X, FS, lam=-1.0)
