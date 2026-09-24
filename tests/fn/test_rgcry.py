"""Tests for rgcry.rangayyan_infant_cry."""

import math
from morie.fn import _array_core as np
from morie.fn.bsaphys import rangayyan_infant_cry


def _synthesise_cry(rng, fs, duration, f0, voiced_start, voiced_end):
    """Build a synthetic infant cry signal with a voiced segment at f0."""
    n = int(duration * fs)
    sig = []
    for i in range(n):
        t = i / fs
        if voiced_start <= t <= voiced_end:
            v = (math.sin(2 * math.pi * f0 * t)
                 + 0.5 * math.sin(2 * math.pi * 2 * f0 * t)
                 + 0.3 * math.sin(2 * math.pi * 3 * f0 * t)
                 + 0.1 * math.sin(2 * math.pi * 4 * f0 * t))
            sig.append(v)
        else:
            sig.append(0.0)
    noise = [0.005 * x for x in rng.normal(0, 1, n)]
    return [s + x for s, x in zip(sig, noise)]


def test_rgcry_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    fs = 8000.0
    cry = _synthesise_cry(rng, fs, duration=1.0, f0=400.0,
                          voiced_start=0.2, voiced_end=0.8)
    result = rangayyan_infant_cry(cry, fs)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_rgcry_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    fs = 8000.0
    cry = _synthesise_cry(rng, fs, duration=0.6, f0=450.0,
                          voiced_start=0.1, voiced_end=0.5)
    result = rangayyan_infant_cry(cry, fs)
    assert isinstance(result, dict)
    assert len(result) > 0
