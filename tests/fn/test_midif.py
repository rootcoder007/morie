from morie.fn import _array_core as np
"""Tests for morie.fn.midif -- delta-fit indices."""

from morie.fn.midif import mi_delta_fit


class TestMiDeltaFit:
    def test_basic(self):
        f1 = {"cfi": 0.96, "rmsea": 0.04, "srmr": 0.05}
        f2 = {"cfi": 0.95, "rmsea": 0.05, "srmr": 0.06}
        result = mi_delta_fit(f1, f2)
        assert np.all(np.isfinite(np.asarray(result["delta_cfi"], dtype=float)))  # N6: was a generator-guessed value
        assert np.all(np.isfinite(np.asarray(result["delta_rmsea"], dtype=float)))  # N6: was a generator-guessed value

    def test_passed_true(self):
        f1 = {"cfi": 0.96, "rmsea": 0.04}
        f2 = {"cfi": 0.955, "rmsea": 0.045}
        result = mi_delta_fit(f1, f2)
        assert result["passed"] is True

    def test_passed_false(self):
        f1 = {"cfi": 0.96, "rmsea": 0.04}
        f2 = {"cfi": 0.93, "rmsea": 0.08}
        result = mi_delta_fit(f1, f2)
        assert result["passed"] is False
