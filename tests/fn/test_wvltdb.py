"""Tests for wvltdb.db_wavelet (periodic Daubechies DWT)."""

import math

import pytest

from morie.fn.wvltdb import db_wavelet


Y = [math.sin(0.37 * k) + 0.3 * math.cos(1.9 * k) for k in range(16)]


def test_wvltdb_basic():
    """db2's scaling filter is (1+s3, 3+s3, 3-s3, 1-s3)/(4 sqrt 2) with
    s3 = sqrt 3 (Daubechies 1992, ch. 6); the transform is orthonormal
    (energy is preserved across levels) and reconstructs exactly."""
    s3 = math.sqrt(3)
    r = db_wavelet(Y, level=3, wavelet="db2")
    assert r["h"] == pytest.approx([(1 + s3) / (4 * math.sqrt(2)), (3 + s3) / (4 * math.sqrt(2)),
                                    (3 - s3) / (4 * math.sqrt(2)), (1 - s3) / (4 * math.sqrt(2))], abs=1e-15)
    tot = sum(v * v for v in Y)
    assert r["approximation_energy"] + sum(r["energies"]) == pytest.approx(tot, rel=1e-13)
    assert r["reconstruction_error"] < 1e-13
    assert [len(d) for d in r["details"]] == [8, 4, 2]


def test_wvltdb_edge():
    """db1 is the Haar transform: first-level details (x_2k - x_2k+1)/sqrt 2
    up to sign, approximations (x_2k + x_2k+1)/sqrt 2; bad lengths raise."""
    r = db_wavelet(Y, level=1, wavelet="db1")
    a = [(Y[2 * k] + Y[2 * k + 1]) / math.sqrt(2) for k in range(8)]
    d = [(Y[2 * k] - Y[2 * k + 1]) / math.sqrt(2) for k in range(8)]
    assert r["approximation"] == pytest.approx(a, abs=1e-14)
    assert [abs(v) for v in r["details"][0]] == pytest.approx([abs(v) for v in d], abs=1e-14)
    with pytest.raises(ValueError):
        db_wavelet(Y[:12])
    with pytest.raises(ValueError):
        db_wavelet(Y, wavelet="db9")


