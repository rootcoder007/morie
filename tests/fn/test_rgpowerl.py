"""Tests for rgpowerl.rangayyan_powerline_removal."""

import math

import pytest

from morie.fn.bsaqrs import rangayyan_powerline_removal


def test_rgpowerl_basic():
    """The book's case: fo = 60 Hz, fs = 1000 Hz gives 1 - 1.85955 z^-1 +
    z^-2 with DC gain 0.14045, divided out so DC passes at unity."""
    r = rangayyan_powerline_removal([0.0] * 10, 1000.0, 60.0)
    b0, b1, b2 = r["coeffs"][0]
    g = 2 - 2 * math.cos(2 * math.pi * 60 / 1000)
    assert round(g, 5) == 0.14045
    assert round(b1 * g, 5) == -1.85955
    assert (b0, b2) == (1 / g, 1 / g)
    assert b0 + b1 + b2 == pytest.approx(1.0, rel=1e-12)


def test_rgpowerl_edge():
    """Zeros on the unit circle null a 60 Hz sinusoid exactly once the two
    start-up samples have passed; a DC level passes unchanged; the comb
    also removes the 180 Hz harmonic."""
    x = [0.7 + math.sin(2 * math.pi * 60 * n / 1000) + 0.3 * math.cos(2 * math.pi * 180 * n / 1000)
         for n in range(200)]
    y = rangayyan_powerline_removal(x, 1000.0, 60.0, harmonics=3)
    assert y["notched"] == [60.0, 120.0, 180.0]
    # each normalised stage has |b0| + |b1| + |b2| near 27, so three
    # stages scale rounding error by about 27^3 eps = 4e-12
    assert max(abs(v - 0.7) for v in y["y"][6:]) < 1e-11
    with pytest.raises(ValueError, match="Nyquist"):
        rangayyan_powerline_removal(x, 100.0, 60.0)


