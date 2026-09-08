"""Tests for tarmd.threshold_autoregression."""

from morie.fn import _array_core as np
import pytest

from morie.fn.tarmd import threshold_autoregression


def _setar(seed, n=2000, c=0.0, phi_lo=0.7, phi_hi=-0.4):
    """SETAR(1): x_t = phi_lo x_{t-1} + e if x_{t-1} <= c else phi_hi x_{t-1} + e
    (Tong 1990)."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        phi = phi_lo if x[t - 1] <= c else phi_hi
        x[t] = phi * x[t - 1] + 0.5 * rng.standard_normal()
    return x


def test_tarmd_recovers_threshold_and_both_regimes():
    """Mean over seeds 1..3: threshold near 0, AR coefficients near
    (0.7, -0.4). Signs differ across regimes, so a linear AR cannot fake
    this."""
    cs, lo, hi = [], [], []
    for s in (1, 2, 3):
        r = threshold_autoregression(_setar(s), p=1, d=1)
        cs.append(float(r["threshold"]))
        lo.append(float(np.asarray(r["phi_lower"], dtype=float).ravel()[-1]))
        hi.append(float(np.asarray(r["phi_upper"], dtype=float).ravel()[-1]))
    assert abs(np.mean(cs)) < 0.25
    assert np.mean(lo) == pytest.approx(0.7, abs=0.15)
    assert np.mean(hi) == pytest.approx(-0.4, abs=0.15)


def test_tarmd_regimes_partition_the_sample():
    x = _setar(5, n=800)
    r = threshold_autoregression(x, p=1, d=1)
    sizes = np.asarray(r["regime_sizes"], dtype=int)
    assert sizes.sum() == len(x) - 1  # one lag consumed
    assert sizes.min() > 50


def test_tarmd_sse_beats_a_linear_ar_fit():
    """The split must reduce the SSE relative to one global AR(1) -- that
    is the whole point of thresholding."""
    x = _setar(6, n=1000)
    r = threshold_autoregression(x, p=1, d=1)
    X, Y = x[:-1], x[1:]
    phi = float(X @ Y) / float(X @ X)
    sse_linear = float(((Y - phi * X) ** 2).sum())
    assert float(r["sse"]) < sse_linear * 0.9


def test_tarmd_rejects_short_series():
    with pytest.raises(ValueError, match="too short"):
        threshold_autoregression(np.arange(8.0), p=3, d=2)
