"""Tests for morie.fn.symtz -- Symmetrization via Rademacher complexity."""

import pytest

from morie.fn import _array_core as np
from morie.fn.symtz import symmetrization_bound


class TestSymmetrization:
    def test_basic_output(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        r = symmetrization_bound(x)
        assert r["symmetrization_bound"] == pytest.approx(2.0 * r["rademacher_avg"])
        assert r["n"] == 100

    def test_zero_data(self):
        x = np.zeros(50)
        r = symmetrization_bound(x)
        assert r["rademacher_avg"] == pytest.approx(0.0, abs=1e-10)

    def test_different_functionals(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        for fn in ("mean", "var", "max", "abs_mean"):
            r = symmetrization_bound(x, fn=fn)
            assert r["rademacher_avg"] >= 0

    def test_unknown_fn(self):
        with pytest.raises(ValueError, match="fn must be"):
            symmetrization_bound(np.array([1, 2, 3]), fn="bad")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            symmetrization_bound(np.array([]))


def test_rademacher_average_replayed():
    from morie.fn import _array_core as np

    x = np.array([0.5, -1.2, 0.3, 2.0, -0.7])
    rng = np.random.default_rng(3)
    vals = []
    for _ in range(40):
        s = rng.choice([-1.0, 1.0], size=5)
        vals.append(abs(float(np.mean(s * x))))
    r = symmetrization_bound(x, n_rademacher=40, seed=3)
    assert r["rademacher_avg"] == pytest.approx(sum(vals) / 40, rel=1e-13)
    assert r["empirical_value"] == pytest.approx(float(np.mean(x)), rel=1e-14)
