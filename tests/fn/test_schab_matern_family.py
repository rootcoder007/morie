"""Book-certified tests for the Matern family (Schabenberger & Gotway 2005).

Every assertion is an identity the book states. The Matern special cases
are the strongest available check: the book names what nu = 1/2 and
nu = 1 must reduce to, so those are exact targets rather than
self-generated numbers.
"""

import numpy as np
import pytest
from scipy.special import kv

from morie.fn.spmatr import schabenberger_matern_covariance as matern
from morie.fn.spbesf import schabenberger_bessel_function as bessel_k
from morie.fn.spbesf import _bessel_i
from morie.fn.spanis import schabenberger_geometric_anisotropy as anisotropy

H = np.array([0.0, 0.3, 1.0, 2.5, 4.0])
THETA, SIGMA2 = 1.7, 2.0


def test_matern_variance_at_the_origin_is_sigma2():
    """K_nu(t) ~ Gamma(nu)/2 (t/2)^-nu as t -> 0, so C(0) = sigma^2 (p. 143)."""
    for nu in (0.25, 0.5, 1.0, 2.0, 5.0):
        assert matern(H, SIGMA2, nu, THETA)["covariance"][0] == pytest.approx(SIGMA2)


def test_matern_at_nu_one_half_is_the_exponential_model():
    """nu = 1/2 gives C(h) = sigma^2 exp(-theta h) (eq 4.11, p. 144)."""
    got = matern(H, SIGMA2, 0.5, THETA)["covariance"]
    np.testing.assert_allclose(got, SIGMA2 * np.exp(-THETA * H), rtol=1e-12)


def test_matern_at_nu_one_is_whittles_model():
    """nu = 1 gives C(h) = sigma^2 theta h K_1(theta h) (eq 4.12, p. 144)."""
    t = THETA * H
    expected = np.where(t > 0, SIGMA2 * t * kv(1, np.where(t > 0, t, 1.0)), SIGMA2)
    np.testing.assert_allclose(matern(H, SIGMA2, 1.0, THETA)["covariance"],
                               expected, rtol=1e-12)


def test_matern_smoothness_increases_with_nu():
    """"the semivariogram rises more quickly from the origin as nu
    decreases" (p. 145, Figure 4.2)."""
    near = np.array([0.05])
    g = [matern(near, 1.0, nu, 1.0)["semivariogram"][0]
         for nu in (0.25, 0.5, 1.0, 3.0)]
    assert g[0] > g[1] > g[2] > g[3]


def test_matern_covariance_decreases_with_lag():
    c = matern(np.array([0.1, 0.5, 1.0, 3.0]), 1.0, 1.5, 1.0)["covariance"]
    assert np.all(np.diff(c) < 0)


def test_matern_rejects_invalid_parameters():
    for kwargs in ({"nu": 0.0}, {"nu": -1.0}, {"a": 0.0}, {"sigma2": 0.0}):
        with pytest.raises(ValueError):
            matern(H, **{"sigma2": 1.0, "nu": 1.0, "a": 1.0, **kwargs})


def test_bessel_k_matches_the_books_own_identity_where_it_is_stable():
    """Eq (4.73): K_nu = (pi/2)(I_-nu - I_nu)/sin(pi nu).

    The identity is a definition, not an algorithm -- I_{+-nu} both grow
    like e^t while the difference decays like e^-t, so it is only usable
    at small argument. Checked where it is trustworthy.
    """
    t = np.array([0.05, 0.2, 0.5, 1.0, 2.0, 4.0])
    for nu in (0.25, 0.5, 1.5, 2.5):
        identity = (np.pi / 2.0) * (_bessel_i(-nu, t) - _bessel_i(nu, t)) \
            / np.sin(np.pi * nu)
        np.testing.assert_allclose(bessel_k(t, nu)["value"], identity, rtol=1e-9)


def test_bessel_k_half_order_closed_form():
    """K_{1/2}(t) = sqrt(pi/2t) e^-t (p. 143)."""
    t = np.array([0.1, 1.0, 5.0, 20.0])
    np.testing.assert_allclose(bessel_k(t, 0.5)["value"],
                               np.sqrt(np.pi / (2 * t)) * np.exp(-t), rtol=1e-12)


def test_bessel_k_diverges_at_the_origin():
    with pytest.raises(ValueError, match="positive"):
        bessel_k(np.array([0.0]), 0.5)


def test_anisotropy_correction_restores_isotropy():
    """C(h) = C1(||Bh||); Z(As) is isotropic again when A = B^-1 (p. 151).

    Build a field that is isotropic in s*, stretch it by B to make it
    anisotropic, then check A = B^-1 recovers the isotropic
    semivariogram.
    """
    rng = np.random.default_rng(7)
    star = rng.random((300, 2)) * 10.0            # isotropic space
    z = np.sin(star[:, 0] * 0.7) + np.cos(star[:, 1] * 0.7)
    B = np.array([[1.0, 0.0], [0.0, 4.0]])        # stretch y fourfold
    observed = star @ np.linalg.inv(B).T          # what we would have measured
    A = B

    r = anisotropy(observed, z, A, n_bins=10)
    ref = anisotropy(star, z, np.eye(2), n_bins=10)
    ok = ~np.isnan(r["gamma"]) & ~np.isnan(ref["gamma"])
    # corrected space reproduces the truly isotropic semivariogram
    np.testing.assert_allclose(r["gamma"][ok], ref["gamma"][ok], rtol=1e-6)
    # and the uncorrected one is measurably different
    assert not np.allclose(r["gamma_raw"][ok], r["gamma"][ok], rtol=1e-3)


def test_anisotropy_identity_matrix_is_a_no_op():
    rng = np.random.default_rng(2)
    coords = rng.random((120, 2))
    z = rng.normal(size=120)
    r = anisotropy(coords, z, np.eye(2), n_bins=6)
    np.testing.assert_allclose(r["coords_corrected"], coords, rtol=1e-12)
    ok = ~np.isnan(r["gamma"])
    np.testing.assert_allclose(r["gamma"][ok], r["gamma_raw"][ok], rtol=1e-12)


def test_anisotropy_rejects_a_singular_map():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="singular"):
        anisotropy(rng.random((20, 2)), rng.normal(size=20),
                   np.array([[1.0, 2.0], [2.0, 4.0]]))
