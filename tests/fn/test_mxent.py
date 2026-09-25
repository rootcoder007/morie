"""Tests for morie.fn.mxent — maximum entropy distribution."""

from morie.fn import _array_core as np
import pytest

from morie.fn.mxent import mxent


class TestMxent:
    def test_no_constraints_uniform(self):
        support = np.arange(5, dtype=float)
        result = mxent(support, [])
        # no constraint: exactly uniform, no optimisation involved
        assert [float(v) for v in result["pmf"]] == [0.2] * 5

    def test_mean_constraint(self):
        # a mean at the centre of a symmetric support is met by the
        # uniform pmf, where the dual gradient c - E f is already 0
        support = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mxent(support, [(lambda x: x, 3.0)])
        actual_mean = np.sum(result["pmf"] * support)
        assert actual_mean == pytest.approx(3.0, rel=1e-15)

    def test_the_solution_is_the_gibbs_family(self):
        """Jaynes: p_i = exp(-lambda f(x_i)) / Z. The pmf must be exactly
        that for the returned multiplier, and the dual gradient -- the
        constraint error -- is what BFGS drives below gtol = 1e-6."""
        import math

        support = np.arange(4, dtype=float)
        result = mxent(support, [(lambda x: x, 1.0)])
        lam = float(result["lagrange_multipliers"][0])
        w = [math.exp(-lam * x) for x in range(4)]
        assert [float(v) for v in result["pmf"]] == pytest.approx([v / sum(w) for v in w], rel=1e-12)
        assert abs(result["constraint_errors"][0]) < 1e-6
        assert lam > 0      # a mean below the centre tilts towards small x

    def test_pmf_sums_to_one(self):
        support = np.arange(4, dtype=float)
        result = mxent(support, [(lambda x: x, 1.5)])
        assert np.sum(result["pmf"]) == pytest.approx(1.0, rel=1e-15)

    def test_entropy_positive(self):
        support = np.arange(3, dtype=float)
        result = mxent(support, [])
        assert result["entropy"] > 0

    def test_constraint_errors_small(self):
        support = np.array([0.0, 1.0, 2.0])
        result = mxent(support, [(lambda x: x, 1.0)])
        # symmetric again: uniform on {0, 1, 2} has mean 1 exactly
        for err in result["constraint_errors"]:
            assert abs(err) < 1e-15

    def test_empty_support_error(self):
        with pytest.raises(ValueError):
            mxent(np.array([]), [])
