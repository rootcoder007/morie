"""Tests for emdvr.py - EMD variance ratio."""

from morie.fn import _array_core as np
from morie.fn.emdvr import emd_variance_ratio, emdvr


def test_variance_ratio_returns_result():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    imfs = [rng.standard_normal(100) for _ in range(3)]
    result = emd_variance_ratio(x, imfs)
    assert result.name == "emd_variance_ratio"
    assert "ratios" in result.extra
    assert len(result.extra["ratios"]) == 3


def test_variance_ratio_sums():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)
    imf1 = x * 0.6
    imf2 = x * 0.4
    result = emd_variance_ratio(x, [imf1, imf2])
    ratios = result.extra["ratios"]
    assert all(r >= 0 for r in ratios)


def test_variance_ratio_alias():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    result = emdvr(x, [x * 0.5])
    assert result.name == "emd_variance_ratio"


def test_variance_ratios_recomputed():
    import pytest

    x = [0.5, 1.2, -0.3, 0.8, 2.0, -1.1]
    imfs = [[0.3, 0.7, -0.2, 0.5, 1.1, -0.6], [0.2, 0.5, -0.1, 0.3, 0.9, -0.5]]

    def v0(a):
        m = sum(a) / len(a)
        return sum((t - m) ** 2 for t in a) / len(a)

    want = [v0(i) / (v0(x) + 1e-12) for i in imfs]
    r = emd_variance_ratio(x, imfs)
    assert [float(v) for v in r.extra["ratios"]] == pytest.approx(want, rel=1e-12)
    assert r.value == pytest.approx(sum(want), rel=1e-12)
