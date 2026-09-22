"""Tests for gh_c7_3.ghosal_exp_dens_kl."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_c7_3 import ghosal_exp_dens_kl


def _psi(x, cs):
    return sum(c * math.cos((k + 1) * math.pi * x) for k, c in enumerate(cs))


def test_gh_c7_3_basic():
    """Test basic functionality with documented coefficient arguments."""
    coefs0 = (0.5, -0.3)
    coefs = (0.45, -0.25)
    n_int = 800
    result = ghosal_exp_dens_kl(coefs0=coefs0, coefs=coefs, n_int=n_int)

    # Payload must contain the documented keys.
    assert "estimate" in result
    assert "sup_norm_gap" in result
    assert "kl_small_when_sup_small" in result
    assert "method" in result

    # estimate is max(kl, 0.0): non-negative and finite.
    estimate = float(result["estimate"])
    assert np.all(np.isfinite(np.asarray(estimate, dtype=float)))
    assert estimate >= 0.0

    # Independently compute sup_norm_gap over the same grid used by the
    # function (n=200 midpoints on [0,1]) and compare.
    pts = [(i + 0.5) / 200 for i in range(200)]
    expected_sup = max(
        abs(_psi(x, list(coefs0)) - _psi(x, list(coefs))) for x in pts
    )
    assert math.isclose(float(result["sup_norm_gap"]), expected_sup,
                        rel_tol=1e-12, abs_tol=1e-12)

    # The "kl_small_when_sup_small" boolean must be self-consistent with
    # estimate and sup_norm_gap as documented.
    expected_flag = estimate <= 2.0 * expected_sup * math.exp(expected_sup)
    assert bool(result["kl_small_when_sup_small"]) is expected_flag


def test_gh_c7_3_edge():
    """Test edge case: identical coefficient vectors yield zero KL and zero sup gap."""
    cs = (0.1, 0.2, -0.3)
    result = ghosal_exp_dens_kl(coefs0=cs, coefs=cs, n_int=200)

    # Documented keys are present; no key 'n' is returned.
    assert "estimate" in result
    assert "sup_norm_gap" in result
    assert "kl_small_when_sup_small" in result
    assert "method" in result

    # When the two expansions coincide, both the sup gap and the (clipped)
    # KL estimate must be (numerically) zero.
    assert math.isclose(float(result["sup_norm_gap"]), 0.0,
                        abs_tol=1e-12)
    assert math.isclose(float(result["estimate"]), 0.0,
                        abs_tol=1e-6)

    # Independent verification of the trivial sup gap.
    pts = [(i + 0.5) / 200 for i in range(200)]
    expected_sup = max(
        abs(_psi(x, list(cs)) - _psi(x, list(cs))) for x in pts
    )
    assert expected_sup == 0.0
