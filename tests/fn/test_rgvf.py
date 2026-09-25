"""Tests for bsaqrs.rangayyan_vf_detect (spectral concentration and QRS absence)."""

import math

import pytest

from morie.fn.bsaqrs import rangayyan_vf_detect


FS = 250.0
VF = [math.sin(2 * math.pi * 5 * t / FS) + 0.05 * math.sin(0.37 * t) for t in range(2000)]
SINUS = [3 * math.exp(-((t % 200) - 100) ** 2 / 8.0) + 0.02 * math.sin(0.3 * t) for t in range(2000)]


def test_rgvf_basic():
    """A window is flagged exactly when its spectral concentration
    reaches `conc` AND its integrator crest factor stays below
    `crest`; `fraction` is the share of flagged windows.  A sustained
    5 Hz oscillation is VF-like, a spiky 75 bpm rhythm is not."""
    for sig, expect in ((VF, True), (SINUS, False)):
        r = rangayyan_vf_detect(sig, FS)
        flags = [c >= r["conc"] and k <= r["crestmax"] for c, k in zip(r["concentration"], r["crest"])]
        assert list(r["flag"]) == flags
        assert r["fraction"] == pytest.approx(sum(flags) / len(flags), abs=1e-15)
        assert all(f is expect for f in flags)
    assert rangayyan_vf_detect(VF, FS)["domfreq"][0] == pytest.approx(5.0, abs=0.25)
    assert rangayyan_vf_detect(SINUS, FS)["rate"][0] == pytest.approx(75.0, abs=1e-9)


def test_rgvf_edge():
    """Windows count as floor(duration / win); a segment shorter than
    one window raises."""
    assert rangayyan_vf_detect(VF, FS, win=2.0)["nwin"] == 4
    with pytest.raises(ValueError):
        rangayyan_vf_detect(VF[:100], FS)
