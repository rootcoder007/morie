"""Tests for wave.wavelet_basis."""

import math

import pytest

from morie.fn.wave import wavelet_basis


def _dot(row, x):
    return sum(a * b for a, b in zip(row, x))


def test_wave_basic():
    """The basis is orthonormal and really is the matrix of the transform."""
    y = [math.sin(0.4 * i) + 0.25 * i for i in range(16)]

    for wavelet, flen in (("db1", 2), ("db2", 4), ("db3", 6)):
        result = wavelet_basis(y, wavelet, level=2)
        assert result["n"] == 16
        assert result["filter_length"] == flen
        assert result["level"] == 2
        h = result["h"]
        g = result["g"]
        assert len(h) == flen

        # Daubechies' defining conditions, recomputed here from h.
        assert sum(v * v for v in h) == pytest.approx(1.0, abs=1e-12)
        assert sum(h) == pytest.approx(math.sqrt(2.0), abs=1e-12)
        for m in range(1, flen // 2):
            lag = sum(h[k] * h[k + 2 * m] for k in range(flen - 2 * m))
            assert lag == pytest.approx(0.0, abs=1e-12)
        for p in range(flen // 2):
            mom = sum((-1.0) ** k * (k ** p if p else 1.0) * h[k]
                      for k in range(flen))
            assert mom == pytest.approx(0.0, abs=1e-12)
        # quadrature mirror g_k = (-1)^k h_{L-1-k}
        for k in range(flen):
            assert g[k] == pytest.approx((-1.0) ** k * h[flen - 1 - k],
                                         abs=1e-15)

        # the payload residuals agree with those recomputations
        assert result["orthonormality"] < 1e-12
        assert result["normalisation"] < 1e-12
        assert result["double_shift"] < 1e-12
        assert result["vanishing_moments"] < 1e-12
        assert result["gram_error"] < 1e-10

        # coefficients are W y, and W is orthonormal so energy is preserved
        W = result["basis"]
        co = result["coefficients"]
        assert len(W) == 16 and len(W[0]) == 16
        for i in range(16):
            assert co[i] == pytest.approx(_dot(W[i], y), abs=1e-12)
        assert result["energy_out"] == pytest.approx(result["energy_in"],
                                                     rel=1e-12)
        assert result["energy_in"] == pytest.approx(
            sum(v * v for v in y), rel=1e-12)
        assert result["estimate"] == pytest.approx(result["energy_out"],
                                                   rel=1e-12)
        # W' W = I, so W' c reconstructs the signal exactly
        for j in range(16):
            rec = sum(W[i][j] * co[i] for i in range(16))
            assert rec == pytest.approx(y[j], abs=1e-10)


def test_wave_haar_one_level_is_sums_and_differences():
    """db1 at level 1 is (a+b)/sqrt2 and (a-b)/sqrt2, by arithmetic."""
    y = [1.0, 2.0, 3.0, 4.0]
    res = wavelet_basis(y, "db1", level=1)
    r2 = math.sqrt(2.0)
    expected = [(1.0 + 2.0) / r2, (3.0 + 4.0) / r2,
                (1.0 - 2.0) / r2, (3.0 - 4.0) / r2]
    for got, want in zip(res["coefficients"], expected):
        assert got == pytest.approx(want, abs=1e-13)
    assert res["level"] == 1

    # full depth on the same signal: the coarsest coefficient is the
    # scaled mean, since Haar averaging applied twice divides by 2.
    full = wavelet_basis(y, "db1")
    assert full["level"] == 2
    assert full["coefficients"][0] == pytest.approx(
        (1.0 + 2.0 + 3.0 + 4.0) / 2.0, abs=1e-13)


def test_wave_edge():
    """Documented failure modes and the shortest admissible signal."""
    # length 2 with db1 is the smallest legal case
    res = wavelet_basis([3.0, -1.0], "db1")
    r2 = math.sqrt(2.0)
    assert res["coefficients"][0] == pytest.approx(2.0 / r2, abs=1e-13)
    assert res["coefficients"][1] == pytest.approx(4.0 / r2, abs=1e-13)
    assert res["n"] == 2 and res["level"] == 1

    with pytest.raises(ValueError):
        wavelet_basis([1.0] * 100, "db2")        # not a power of two
    with pytest.raises(ValueError):
        wavelet_basis([1.0] * 16, "morl")        # unknown wavelet
    with pytest.raises(ValueError):
        wavelet_basis([1.0] * 16, "db2", level=0)
    with pytest.raises(ValueError):
        wavelet_basis([1.0] * 16, "db2", level=5)
    with pytest.raises(ValueError):
        # db3 has L = 6; at level 4 the coarsest stage has only 2 samples
        wavelet_basis([1.0] * 16, "db3", level=4)

    # the full default depth is only admissible while the filter still
    # fits the coarsest stage: db1 reaches log2(16) = 4 levels, db2 does not.
    assert wavelet_basis([1.0] * 16, "db1")["level"] == 4
    with pytest.raises(ValueError):
        wavelet_basis([1.0] * 16, "db2")
