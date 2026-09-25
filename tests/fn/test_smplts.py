"""Tests for smplts.sample_lifetable (actuarial life table)."""

import pytest

from morie.fn.smplts import sample_lifetable


def test_smplts_basic():
    """q_j = d_j / (n_j - w_j / 2), S_j = prod_{k <= j} (1 - q_k)."""
    iv = [0, 1, 2, 3, 4]
    n, d, w = [100, 80, 60, 35], [12, 10, 15, 6], [8, 10, 10, 29]
    r = sample_lifetable(iv, n, d, w)
    q = [dj / (nj - wj / 2) for dj, nj, wj in zip(d, n, w)]
    S, acc = [], 1.0
    for v in q:
        acc *= 1 - v
        S.append(acc)
    assert list(r["q"]) == pytest.approx(q, rel=1e-15)
    assert list(r["survival"]) == pytest.approx(S, rel=1e-14)
    assert list(r["effective_n"]) == pytest.approx([nj - wj / 2 for nj, wj in zip(n, w)], rel=1e-15)


def test_smplts_edge():
    """Without withdrawals q is the plain death fraction; boundaries that
    are not increasing are refused."""
    r = sample_lifetable([0, 5, 10], [50, 40], [10, 8])
    assert list(r["q"]) == [0.2, 0.2]
    with pytest.raises(ValueError):
        sample_lifetable([0, 5, 5], [50, 40], [10, 8])
