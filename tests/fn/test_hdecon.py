"""Tests for hdecon — Homomorphic deconvolution."""

from morie.fn import _array_core as np

from morie.fn._containers import SignalResult
from morie.fn.hdecon import homomorphic_deconvolve


def test_hdecon_basic(rng):
    x = rng.standard_normal(256)
    result = homomorphic_deconvolve(x, cutoff=30)
    assert isinstance(result, SignalResult)
    assert result.filtered is not None
    assert "excitation" in result.extra


def test_hdecon_output_length():
    x = np.random.default_rng(0).standard_normal(100)
    result = homomorphic_deconvolve(x, cutoff=20)
    assert len(result.filtered) == 100
    assert len(result.extra["excitation"]) == 100


def test_hdecon_components_reconvolve_to_the_signal():
    """The two lifters split the complex cepstrum exactly, so log H + log E
    = log X and H E = X: with n = n_fft the circular convolution of the
    minimum-phase part and the excitation rebuilds the input."""
    import math
    n = 64
    x = [math.exp(-0.2 * t) * math.cos(0.9 * t) + (0.6 if t % 16 == 3 else 0.0) for t in range(n)]
    r = homomorphic_deconvolve(x, cutoff=10)
    h = list(r.filtered)
    e = list(r.extra["excitation"])
    rebuilt = [sum(h[k] * e[(t - k) % n] for k in range(n)) for t in range(n)]
    assert max(abs(a - b) for a, b in zip(rebuilt, x)) < 1e-9
