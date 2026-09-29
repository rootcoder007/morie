"""Tests for morie.fn.cusum — CUSUM change detection."""

from morie.fn import _array_core as np
from morie.fn.cusum import cusum


class TestCusum:
    """Tests for cusum()."""

    def test_detects_mean_shift(self):
        """Detects a clear mean shift."""
        series = np.concatenate(
            [
                np.zeros(50),
                np.full(50, 5.0),
            ]
        )
        result = cusum(series, target_mean=0.0, threshold=10.0)
        assert len(result["change_points"]) > 0
        # Change point should be near index 50
        assert any(45 <= cp <= 60 for cp in result["change_points"])

    def test_no_change_in_constant(self):
        """No change detected in a constant series."""
        series = np.full(100, 3.0)
        result = cusum(series, threshold=5.0)
        assert len(result["change_points"]) == 0

    def test_arrays_correct_length(self):
        """CUSUM arrays match series length."""
        series = np.random.default_rng(42).standard_normal(80)
        result = cusum(series)
        assert len(result["cusum_pos"]) == 80
        assert len(result["cusum_neg"]) == 80


def test_page_cusum_paths_recomputed():
    import pytest

    x = [0.1, -0.2, 0.3, 1.5, 1.8, 2.1, 0.2, -0.1]
    mu0, k, h = 0.0, 0.2, 2.0
    sp, sn, cps, P, N = 0.0, 0.0, [], [], []
    for t, v in enumerate(x):
        sp = max(0.0, sp + v - mu0 - k)
        sn = max(0.0, sn - (v - mu0) - k)
        if sp > h or sn > h:
            cps.append(t)
            sp = sn = 0.0
        P.append(sp)
        N.append(sn)
    r = cusum(x, target_mean=mu0, threshold=h, drift=k)
    assert [float(v) for v in r["cusum_pos"]] == pytest.approx(P, rel=1e-14, abs=1e-15)
    assert [float(v) for v in r["cusum_neg"]] == pytest.approx(N, rel=1e-14, abs=1e-15)
    assert r["change_points"] == cps
