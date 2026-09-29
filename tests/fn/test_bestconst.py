"""Tests for bestconst (Morin 2016 eqs 6.22-6.23)."""

import pytest

from morie.fn.bestconst import bestconst


def test_bestconst_is_the_mean_with_its_mse():
    y = [2.0, 4.5, 3.0, 7.5, 1.0]
    m = sum(y) / 5
    r = bestconst(y)
    assert r["best_prediction"] == pytest.approx(m, rel=1e-14)
    assert r["mse"] == pytest.approx(sum((v - m) ** 2 for v in y) / 5, rel=1e-13)
    # any other constant does worse
    for c in (m - 0.1, m + 0.1):
        assert sum((v - c) ** 2 for v in y) / 5 > r["mse"]
