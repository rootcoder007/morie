"""Tests for rgdaub.rangayyan_daubechies."""

import math

import pytest

from morie.fn.bsatf import rangayyan_daubechies


def test_rgdaub_basic():
    """db2 equals its closed form (1 + sqrt 3, 3 + sqrt 3, 3 - sqrt 3,
    1 - sqrt 3) / (4 sqrt 2); the analysis filters are the reversed
    synthesis filters (PyWavelets' convention)."""
    s3, r2 = math.sqrt(3), math.sqrt(2)
    r = rangayyan_daubechies(2)
    exact = [(1 + s3) / (4 * r2), (3 + s3) / (4 * r2), (3 - s3) / (4 * r2), (1 - s3) / (4 * r2)]
    assert r["rec_lo"] == pytest.approx(exact, rel=1e-15)
    assert r["dec_lo"] == list(reversed(r["rec_lo"]))
    assert r["dec_hi"] == list(reversed(r["rec_hi"]))
    assert r["rec_hi"] == [((-1) ** n) * r["rec_lo"][3 - n] for n in range(4)]


def test_rgdaub_edge():
    """Every order satisfies sum h = sqrt 2, sum h^2 = 1 and the double-shift
    orthogonality to machine precision; Haar is (1, 1)/sqrt 2."""
    for k in range(1, 11):
        r = rangayyan_daubechies(k)
        assert r["length"] == 2 * k
        assert r["sum_lo"] == pytest.approx(math.sqrt(2), abs=1e-14)
        assert r["norm_lo"] == pytest.approx(1.0, abs=1e-14)
        assert r["max_shift_inner_product"] < 1e-15
    assert rangayyan_daubechies(1)["rec_lo"] == pytest.approx([1 / math.sqrt(2)] * 2, rel=1e-15)
    with pytest.raises(ValueError, match="1..10"):
        rangayyan_daubechies(11)


