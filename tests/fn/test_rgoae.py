"""Tests for bsaphys.rangayyan_oae (OAE spectral measures)."""

import math

import pytest

from morie.fn.bsaphys import rangayyan_freq_domain_feat, rangayyan_oae


FS = 8000.0
X = [math.sin(2 * math.pi * 1500 * t / FS) + 0.5 * math.sin(2 * math.pi * 3000 * t / FS)
     + 0.05 * math.sin(9.1 * t) for t in range(2048)]


def test_rgoae_basic():
    """The PSD moments are the Section 6.4.1 ones computed by the shared
    PSD routine; the dominant emission is the 1.5 kHz line; RMS is the
    root mean square of the segment."""
    r = rangayyan_oae(X, FS)
    ref = rangayyan_freq_domain_feat(X, FS)
    for k in ("total_power", "mean_freq_hz", "median_freq_hz", "fm2_hz2"):
        assert r[k] == pytest.approx(ref[k], rel=1e-12)
    assert r["dominant_freq_hz"] == pytest.approx(1500.0, abs=FS / 2048)
    assert r["rms"] == pytest.approx(math.sqrt(sum(v * v for v in X) / len(X)), rel=1e-6)


def test_rgoae_edge():
    """Too low a sampling rate for the default OAE bands raises."""
    with pytest.raises(ValueError):
        rangayyan_oae(X, 500.0)
