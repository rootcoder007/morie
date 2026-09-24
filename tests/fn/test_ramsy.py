"""Tests for ramsey_reset_test."""

import math

from morie.fn import _array_core as np

from morie.fn.ramsy import ramsey_reset_test


class TestRamseyRESET:
    def test_linear_ok(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (60, 1))
        y = 3.0 * X.ravel() + rng.normal(0, 0.5, 60)
        r = ramsey_reset_test(X, y)
        assert r.test_name == "Ramsey RESET"
        assert math.isfinite(r.p_value)
        assert 0.0 <= r.p_value <= 1.0
        assert r.p_value > 0.05

    def test_nonlinear(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(-2, 2, (60, 1))
        y = X.ravel() * X.ravel() + rng.normal(0, 0.1, 60)
        r = ramsey_reset_test(X, y)
        assert math.isfinite(r.p_value)
        assert 0.0 <= r.p_value <= 1.0
        assert r.p_value < 0.05
