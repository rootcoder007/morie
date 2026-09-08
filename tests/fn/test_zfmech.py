"""zfmech: zero-concentrated differential privacy (Bun & Steinke 2016).

The generated test imported `z_dp_mechanism`, a name this module never
exported, and asserted an "estimate" key the payload never had. Rewritten
against the real API, anchored on the paper's closed forms.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.zfmech import (
    zero_concentrated_dp,
    sigma_for_rho,
    compose,
    group_privacy,
)


def test_rho_is_the_closed_form_of_proposition_1_6():
    """rho = Delta^2 / (2 sigma^2), exactly."""
    for delta, sigma in ((1.0, 1.0), (2.0, 3.0), (0.5, 0.25)):
        assert zero_concentrated_dp(delta, sigma) == pytest.approx(
            delta ** 2 / (2 * sigma ** 2))


def test_sigma_for_rho_inverts_the_mechanism():
    """sigma_for_rho is the inverse of the rho calculation."""
    for delta, rho in ((1.0, 0.5), (3.0, 0.1)):
        sigma = sigma_for_rho(delta, rho)
        assert zero_concentrated_dp(delta, sigma) == pytest.approx(rho)


def test_composition_is_additive_in_rho():
    """zCDP composes by adding rho -- the property it exists for."""
    assert compose([0.1, 0.2, 0.05])["rho"] == pytest.approx(0.35)
    assert compose([0.1, 0.2, 0.05])["k"] == 3


def test_group_privacy_scales_quadratically():
    """A group of k costs k^2 rho, not k rho."""
    rho = 0.01
    assert group_privacy(rho, 3)["rho"] == pytest.approx(9 * rho)
    assert group_privacy(rho, 1)["rho"] == pytest.approx(rho)
