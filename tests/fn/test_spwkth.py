"""Tests for spwkth.schabenberger_wiener_khinchin."""

import math

from morie.fn import _array_core as np

from morie.fn.spwkth import schabenberger_wiener_khinchin

# A Gaussian covariance is its own transform pair, which makes every
# returned number checkable in closed form:
#     C(h) = exp(-h^2 / 2)   <->   s(w) = exp(-w^2 / 2) / sqrt(2 pi)
# The grid is kept small on purpose: the quadrature is a Python loop over
# a hard-coded 20001-node omega grid, so a wide h grid costs minutes.
OMEGA = [0.0, 1.0, 2.0, 3.0]
_RESULT = None


def _gaussian_cov(h):
    return np.exp(-np.asarray(h, dtype=float) ** 2 / 2.0)


def _result():
    global _RESULT
    if _RESULT is None:
        _RESULT = schabenberger_wiener_khinchin(_gaussian_cov, omega=OMEGA,
                                                h_max=8.0, n=81)
    return _RESULT


def test_spwkth_basic():
    """The spectral density of a Gaussian covariance is a scaled Gaussian."""
    result = _result()

    got = [float(v) for v in np.asarray(result["spectral_density"]).ravel()]
    assert len(got) == len(OMEGA)
    for w, s in zip(OMEGA, got):
        want = math.exp(-w * w / 2.0) / math.sqrt(2.0 * math.pi)
        assert abs(s - want) < 1e-12

    # The returned omega grid is the one that was asked for.
    assert [float(v) for v in np.asarray(result["omega"]).ravel()] == OMEGA

    # variance is C(0) = 1, and the density must integrate back to it.
    assert abs(result["variance"] - 1.0) < 1e-12
    assert abs(result["integrated_density"] - 1.0) < 1e-9

    # Nyquist frequency of the h grid: pi / (2 dh), dh = 16 / 80.
    assert abs(result["nyquist_omega"] - math.pi / (2.0 * 0.2)) < 1e-12


def test_spwkth_edge():
    """A non-callable covariance is rejected; the density is even in omega."""
    try:
        schabenberger_wiener_khinchin(np.array([1.0, 2.0, 3.0]))
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for a non-callable cov_func")

    result = _result()
    s = [float(v) for v in np.asarray(result["spectral_density"]).ravel()]
    # s is even, so evaluating at -omega must reproduce the same values.
    mirror = schabenberger_wiener_khinchin(_gaussian_cov,
                                           omega=[-w for w in OMEGA],
                                           h_max=8.0, n=41)
    for a, b in zip(s, [float(v) for v in
                        np.asarray(mirror["spectral_density"]).ravel()]):
        assert abs(a - b) < 1e-12
    # A density is non-negative and peaks at omega = 0.
    assert s[0] == max(s) > 0.0
