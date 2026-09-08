"""mtr2sx: inverse-variance weighted Mendelian randomization.

The generated test imported `sex_specific_mr`, which does not exist.
Rewritten against mendelian_randomization_ivw and anchored on the IVW
closed form rather than on a fabricated payload key.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.mtr2sx import mendelian_randomization_ivw, ratio_estimates


def test_ivw_matches_the_closed_form():
    """theta = sum(w_j beta_yj beta_xj) / sum(w_j beta_xj^2), w = 1/se_y^2."""
    bx = [0.20, 0.35, 0.11, 0.42]
    by = [0.10, 0.18, 0.05, 0.21]
    sy = [0.02, 0.03, 0.02, 0.04]
    sx = [0.01, 0.01, 0.01, 0.01]
    r = mendelian_randomization_ivw(bx, sx, by, sy)
    num = sum(b * a / s ** 2 for a, b, s in zip(bx, by, sy))
    den = sum(a * a / s ** 2 for a, s in zip(bx, sy))
    assert r["estimate"] == pytest.approx(num / den)


def test_exact_proportionality_recovers_the_ratio():
    """If every by_j is exactly c * bx_j, the causal estimate is c."""
    bx = [0.2, 0.4, 0.6]
    c = 0.75
    by = [c * b for b in bx]
    r = mendelian_randomization_ivw(bx, [0.01] * 3, by, [0.02] * 3)
    assert r["estimate"] == pytest.approx(c)


def test_ratio_estimates_are_per_variant_wald_ratios():
    ratios = ratio_estimates([0.2, 0.5], [0.1, 0.4])
    assert list(np.asarray(ratios)) == pytest.approx([0.5, 0.8])
