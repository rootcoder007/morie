"""Tests for morie.fn.ltab -- Abridged life table."""

import pytest

from morie.fn.ltab import life_table


class TestLifeTable:
    def test_lx_starts_at_radix(self):
        result = life_table([0, 5, 10], [100, 50, 200], [10000, 9000, 8000], radix=100000)
        assert result["lx"][0] == 100000

    def test_lx_monotone_decreasing(self):
        result = life_table([0, 1, 5, 10], [50, 20, 30, 100], [5000, 4000, 3500, 3000])
        lx = result["lx"]
        for i in range(len(lx) - 1):
            assert lx[i] >= lx[i + 1]

    def test_last_qx_is_one(self):
        result = life_table([0, 10, 20], [10, 20, 50], [1000, 800, 500])
        assert result["nqx"][-1] == 1.0

    def test_ex_positive(self):
        result = life_table([0, 5, 10, 15], [5, 3, 4, 10], [1000, 900, 800, 600])
        assert all(e > 0 for e in result["ex"])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            life_table([0, 5], [10], [100, 90])


def test_life_table_columns_recomputed():
    a = [0.0, 1.0, 5.0]
    d = [50.0, 20.0, 400.0]
    N = [5000.0, 19000.0, 8000.0]
    n = [1.0, 4.0, 5.0]
    M = [di / Ni for di, Ni in zip(d, N)]
    ax = [0.1, 2.0, 2.5]
    q = [n[i] * M[i] / (1 + (n[i] - ax[i]) * M[i]) for i in range(2)] + [1.0]
    lx = [100000.0]
    for i in range(2):
        lx.append(lx[i] * (1 - q[i]))
    dx = [lx[i] * q[i] for i in range(3)]
    L = [n[i] * lx[i + 1] + ax[i] * dx[i] for i in range(2)] + [lx[2] / M[2]]
    T = [sum(L[i:]) for i in range(3)]
    r = life_table(a, d, N)
    assert r["nqx"] == pytest.approx(q, rel=1e-13)
    assert r["lx"] == pytest.approx(lx, rel=1e-13)
    assert r["nLx"] == pytest.approx(L, rel=1e-13)
    assert r["ex"] == pytest.approx([T[i] / lx[i] for i in range(3)], rel=1e-13)
