"""Tests for spdjkr (Schabenberger Sec 5.6.4) and spmidw (Bivand Sec 8.3.1).

Disjunctive kriging is checked against the book's own Example 5.12 identities
and against Parseval, not against its own output. IDW is checked against the
three properties its source states in prose.
"""

import numpy as np
import pytest

from morie.fn._schab_hermite import (disjunctive_kriging, gauss_hermite,
                                     hermite_coefficients, hermite_e,
                                     hermite_orthonormal,
                                     indicator_coefficients,
                                     standard_normal_cdf)
from morie.fn.spdjkr import schabenberger_disjunctive_kriging as spdjkr
from morie.fn.spmidw import schabenberger_idw as spmidw


def _sites(k=5):
    g = np.arange(float(k))
    return np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)


def _y(coords):
    return np.sin(coords[:, 0] * 0.6) + np.cos(coords[:, 1] * 0.4)


def _rho(h):
    return np.exp(-np.asarray(h, dtype=float) / 3.0)


def test_hermite_recurrence_matches_the_closed_forms():
    """The recurrence H_{p+1} = x H_p - p H_{p-1} must reproduce the
    polynomials the definition (5.64) gives."""
    x = np.array([0.0, 0.5, 1.0, -1.3, 2.7])
    h = hermite_e(x, 4)
    assert np.allclose(h[0], 1.0)
    assert np.allclose(h[1], x)
    assert np.allclose(h[2], x**2 - 1.0)
    assert np.allclose(h[3], x**3 - 3.0 * x)
    assert np.allclose(h[4], x**4 - 6.0 * x**2 + 3.0)


def test_the_standardised_system_is_orthonormal():
    """E[eta_p eta_m] = delta_pm -- the property the whole method rests on,
    and the reason H_p is divided by sqrt(p!) rather than used raw."""
    nodes, weights = gauss_hermite(40)
    eta = hermite_orthonormal(nodes, 5)
    gram = (eta * weights) @ eta.T
    assert np.abs(gram - np.eye(6)).max() < 1e-12
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)


def test_example_5_12_identity_expansion():
    """The book works this one out: for g(Z) = Z the Hermite expansion is
    just H_1, so b_0 = 0, b_1 = 1 and every higher coefficient vanishes."""
    b = hermite_coefficients(lambda v: v, 6)
    assert b[0] == pytest.approx(0.0, abs=1e-12)
    assert b[1] == pytest.approx(1.0, abs=1e-12)
    assert np.abs(b[2:]).max() < 1e-12


def test_indicator_coefficients_use_the_closed_form_not_quadrature():
    """eq (5.72) gives b_0 = F(z_k) exactly. Gauss-Hermite quadrature is
    exact for polynomials and the indicator is a step function, so it
    converges slowly and silently -- and the indicator is the canonical
    target of the method, so that error would land where it matters most."""
    z_k = 0.7
    exact = indicator_coefficients(z_k, 6)
    assert exact[0] == pytest.approx(float(standard_normal_cdf(z_k)), abs=1e-14)
    quad = hermite_coefficients(lambda v: (v <= z_k).astype(float), 6)
    assert abs(quad[0] - exact[0]) > 1e-3       # the failure being guarded


def test_indicator_coefficients_satisfy_parseval():
    """I^2 = I, so sum_p b_p^2 = E[I] = F(z_k). The partial sums must climb
    monotonically toward it -- slowly, because a step function has no fast
    Hermite expansion, which is why the text advises only a few terms."""
    z_k = 0.7
    target = float(standard_normal_cdf(z_k))
    sums = [float((indicator_coefficients(z_k, p) ** 2).sum())
            for p in (6, 14, 40, 120)]
    assert all(s < target for s in sums)
    assert sums == sorted(sums)
    assert sums[-1] > 0.98 * sums[0]


def test_indicator_at_an_extreme_threshold_is_degenerate():
    """z_k far into the upper tail makes the indicator almost surely 1, so
    b_0 -> 1 and every other coefficient -> 0."""
    b = indicator_coefficients(8.0, 6)
    assert b[0] == pytest.approx(1.0, abs=1e-12)
    assert np.abs(b[1:]).max() < 1e-10


def test_disjunctive_component_variances_are_bounded():
    """eq (5.69): sigma^2_eta = 1 - lambda'rho. Since eta_p has unit
    variance, every component variance must lie in [0, 1]."""
    coords = _sites()
    _, _, _, comp = disjunctive_kriging(coords, _y(coords),
                                        np.array([2.3, 1.7]), _rho,
                                        lambda v: v, degree=6)
    assert np.all(comp[1:] >= -1e-12)
    assert np.all(comp[1:] <= 1.0 + 1e-12)


def test_disjunctive_kriging_runs_for_both_targets():
    coords = _sites()
    y = _y(coords)
    a = spdjkr(coords, y, np.array([2.3, 1.7]), cov_model=_rho, degree=8)
    b = spdjkr(coords, y, np.array([2.3, 1.7]), cov_model=_rho, degree=8,
               indicator_threshold=0.7)
    assert np.isfinite(a["prediction"]) and a["variance"] >= 0.0
    assert b["coefficients"][0] == pytest.approx(
        float(standard_normal_cdf(0.7)), abs=1e-14)
    assert 0.0 <= b["prediction"] <= 1.5


# ------------------------------------------------------------------- IDW ---

def test_idw_is_exact_at_an_observation():
    """Bivand Sec 8.3.1: "If s0 coincides with an observation location, the
    observed value is returned to avoid infinite weights." An explicit rule,
    not a limit."""
    coords = _sites()
    z = np.exp(_y(coords))
    res = spmidw(coords, z, coords[3])
    assert res["prediction"] == pytest.approx(z[3], abs=1e-12)
    assert res["exact_hits"]


def test_idw_weights_are_normalised():
    coords = _sites()
    res = spmidw(coords, np.exp(_y(coords)), np.array([2.3, 1.7]))
    assert res["weights"].sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(res["weights"] > 0.0)


def test_idw_converges_to_nearest_neighbour_for_large_power():
    """"for large values IDW converges to the one-nearest-neighbour
    interpolation"."""
    coords = _sites()
    z = np.exp(_y(coords))
    target = np.array([2.3, 1.7])
    nearest = z[int(np.argmin(np.linalg.norm(coords - target, axis=1)))]
    assert spmidw(coords, z, target, power=60.0)["prediction"] == \
        pytest.approx(nearest, rel=1e-8)


def test_idw_with_zero_power_is_the_unweighted_mean():
    """p = 0 makes every weight 1; the correct limit, not a special case."""
    coords = _sites()
    z = np.exp(_y(coords))
    assert spmidw(coords, z, np.array([2.3, 1.7]), power=0.0)["prediction"] == \
        pytest.approx(z.mean(), abs=1e-12)


def test_idw_reports_no_prediction_variance():
    """"inverse distance does not provide prediction error variances" -- the
    reference implementation returns NA. Returning None is honest; inventing
    a number would not be."""
    coords = _sites()
    res = spmidw(coords, np.exp(_y(coords)), np.array([2.3, 1.7]))
    assert res["variance"] is None


def test_idw_rejects_bad_input():
    coords = _sites(3)
    z = np.exp(_y(coords))
    with pytest.raises(ValueError):
        spmidw(coords, z, np.array([1.0, 1.0]), power=-1.0)
    with pytest.raises(ValueError):
        spmidw(coords, z[:-1], np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        spmidw(coords, z, np.array([1.0, 1.0, 1.0]))
