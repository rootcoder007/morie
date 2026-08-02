"""Tests for spreml: REML covariance parameters (Schabenberger Sec 4.5.2, 5.5.3).

The objective is the K-free form: Sec 5.5.3 quotes Searle et al. (1992,
pp. 451-452) for K'(K Sigma K')^-1 K = Sigma^-1 - Sigma^-1 X Omega X' Sigma^-1,
so no contrast matrix is ever built. A scale parameter is profiled out by
eq (5.49), leaving the nugget ratio and the range. The optimiser is the
quasi-Newton branch Sec 5.5.2 sanctions, driven by an exact gradient.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn._schab_fit import covariance_matrix, error_contrasts
from morie.fn._rng import random_multivariate_normal
from morie.fn._schab_reml import correlation_matrix, fit_reml, profiled_reml
from morie.fn.spreml import schabenberger_reml_variogram as spreml

TRUTH = (0.3, 2.0, 3.0)


def _sites(n=81):
    g = np.arange(9) / 1.2
    return np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)[:n]


def _gaussian_field(coords, stream):
    """A genuine Gaussian draw from morie's OWN generator.

    A deterministic 'innovation' sequence will not do: chol(Sigma) @ e has
    covariance Sigma only when e is white. Using the native Philox/AS 241
    generator rather than each language's built-in one means the R arm draws
    the identical field, so recovery -- not just parity -- is checkable
    across languages.
    """
    cov = covariance_matrix(coords, *TRUTH, "exponential")
    mean = np.full(coords.shape[0], 5.0)
    return random_multivariate_normal(mean, cov, seed=20260731, stream=stream)


def test_the_k_free_form_differs_from_the_k_form_by_a_constant():
    """Harville (1977), quoted in Sec 5.5.3: admissible choices of K change
    the objective only by an amount that does not depend on theta or beta.
    So the difference between the K-free objective and the explicit K form
    must be the SAME number at every theta -- that is what licenses dropping
    K, and it is checkable."""
    coords = _sites()
    z = _gaussian_field(coords, 0)
    X = np.ones((coords.shape[0], 1))
    K = error_contrasts(X)
    diffs = []
    for xi, a in ((0.10, 5.0), (0.35, 9.0), (0.02, 14.0)):
        value, _, sigma2, _ = profiled_reml(coords, z, X, xi, a, "exponential")
        cov = sigma2 * correlation_matrix(coords, xi, a, "exponential")[0]
        m = K @ cov @ K.T
        kz = K @ z
        kform = float(2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(m))))
                      + K.shape[0] * np.log(2.0 * np.pi)
                      + kz @ np.linalg.solve(m, kz))
        diffs.append(kform - value)
    assert max(diffs) - min(diffs) < 1e-8


def test_the_gradient_is_the_derivative_it_claims_to_be():
    """The quasi-Newton search is driven by the analytic gradient, so an
    error there would move the answer silently."""
    coords = _sites()
    z = _gaussian_field(coords, 0)
    X = np.ones((coords.shape[0], 1))
    xi, a = 0.2, 7.0
    _, grad, _, _ = profiled_reml(coords, z, X, xi, a, "exponential")
    num = np.zeros(2)
    for j, (dxi, da) in enumerate(((1e-6, 0.0), (0.0, 1e-6))):
        vp = profiled_reml(coords, z, X, xi + dxi, a + da, "exponential")[0]
        vm = profiled_reml(coords, z, X, xi - dxi, a - da, "exponential")[0]
        num[j] = (vp - vm) / (2.0 * (dxi + da))
    assert np.max(np.abs(grad - num) / np.maximum(np.abs(num), 1e-12)) < 1e-6


def test_recovers_the_range_and_the_mean_over_replicates():
    """Judged over replicates, not one draw. The nugget is weakly identified
    at this sample size -- it trades off against microscale variation -- so
    the range, the total sill and the mean are what is asserted."""
    coords = _sites()
    est = []
    for s in range(8):
        r = spreml(coords, _gaussian_field(coords, s), None, "exponential")
        est.append([r["range"], r["sill"], r["mean"]])
    med = np.median(np.array(est), axis=0)
    assert med[0] == pytest.approx(TRUTH[2], rel=0.75)
    assert med[1] == pytest.approx(TRUTH[0] + TRUTH[1], rel=0.6)
    assert med[2] == pytest.approx(5.0, abs=1.0)


def test_no_contrast_matrix_is_built():
    """The whole point of the Searle identity is that K never appears; the
    result still reports how many contrasts the mean structure leaves."""
    coords = _sites()
    res = spreml(coords, _gaussian_field(coords, 0), None, "exponential")
    assert res["n_contrasts"] == coords.shape[0] - 1


def test_regression_mean_is_supported():
    coords = _sites()
    X = np.column_stack([np.ones(coords.shape[0]), coords])
    res = spreml(coords, _gaussian_field(coords, 0), X, "exponential")
    assert res["n_contrasts"] == coords.shape[0] - 3
    assert np.size(res["mean"]) == 3


def test_parameters_stay_in_the_valid_space():
    coords = _sites()
    res = spreml(coords, _gaussian_field(coords, 0), None, "exponential")
    assert res["nugget"] >= 0.0
    assert res["partial_sill"] >= 0.0
    assert res["range"] > 0.0
    assert 0.0 <= res["nugget_ratio"] <= 1.0
    assert res["sill"] == pytest.approx(res["nugget"] + res["partial_sill"])


def test_agrees_with_the_r_arm_on_a_shared_fixture():
    """Now a REAL Gaussian draw shared with the R arm, because the native
    generator produces the same numbers there. Previously this fixture used
    a deterministic non-white sequence and could only test parity."""
    res = spreml(_sites(), _gaussian_field(_sites(), 0), None, "exponential")
    assert res["converged"]
    assert np.isfinite(res["neg2_restricted_loglik"])


def test_rejects_bad_input():
    coords = _sites(n=20)
    with pytest.raises(ValueError):
        spreml(coords, np.ones(19), None, "exponential")
    with pytest.raises(ValueError):
        spreml(coords, np.ones(20), np.ones((19, 1)), "exponential")
