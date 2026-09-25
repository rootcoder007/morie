"""Tests for emdsg -- Empirical Mode Decomposition."""

from morie.fn import _array_core as np

from morie.fn._containers import DescriptiveResult
from morie.fn.bsaphys import emd


def test_emd_basic():
    rng = np.random.default_rng(42)
    fs = 500
    t = np.arange(0, 1.0, 1 / fs)
    x = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t)
    x += rng.standard_normal(len(t)) * 0.05
    result = emd(x, max_imfs=5)
    assert isinstance(result, DescriptiveResult)
    assert "imfs" in result.extra
    assert "residue" in result.extra
    assert "n_imfs" in result.extra
    assert "sift_counts" in result.extra
    assert "is_imf" in result.extra


def test_emd_reconstruction():
    rng = np.random.default_rng(7)
    fs = 200
    t = np.arange(0, 2.0, 1 / fs)
    x = np.sin(2 * np.pi * 5 * t) + 0.3 * np.sin(2 * np.pi * 40 * t)
    result = emd(x, max_imfs=10)
    imfs = result.extra["imfs"]
    residue = result.extra["residue"]
    reconstructed = sum(imfs) + residue
    assert np.allclose(x, reconstructed, atol=1e-6)


def test_emd_imf_count():
    rng = np.random.default_rng(99)
    x = rng.standard_normal(256)
    result = emd(x, max_imfs=4)
    assert result.extra["n_imfs"] <= 4
    assert len(result.extra["imfs"]) == result.extra["n_imfs"]
    assert len(result.extra["sift_counts"]) == result.extra["n_imfs"]
    assert len(result.extra["is_imf"]) == result.extra["n_imfs"]


def test_emd_pure_sine():
    fs = 100
    t = np.arange(0, 1.0, 1 / fs)
    x = np.sin(2 * np.pi * 5 * t)
    result = emd(x, max_imfs=3)
    assert result.extra["n_imfs"] >= 1


def test_emd_imfs_and_residue_sum_to_the_signal():
    """Sifting subtracts each IMF from what remains, so the IMFs plus the
    residue reconstruct the input exactly."""
    import math
    t = [k / 500 for k in range(500)]
    x = [math.sin(2 * math.pi * 10 * s) + 0.5 * math.sin(2 * math.pi * 50 * s) + 0.3 * s for s in t]
    r = emd(x, max_imfs=5)
    imfs = [list(v) for v in (r.extra["imfs"].tolist() if hasattr(r.extra["imfs"], "tolist") else r.extra["imfs"])]
    res = list(r.extra["residue"].tolist() if hasattr(r.extra["residue"], "tolist") else r.extra["residue"])
    rebuilt = [sum(imf[k] for imf in imfs) + res[k] for k in range(500)]
    assert max(abs(a - b) for a, b in zip(rebuilt, x)) < 1e-12
