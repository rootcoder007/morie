"""Verification tests for gh_c11_9.

Ghosal and van der Vaart (2017), eq. (11.3)-(11.4), Bochner's spectral representation.
"""

import math

import pytest

from morie.fn.gh_c11_9 import ghosal_statgp_spec


def test_bochners_theorem_closes_for_the_square_exponential_kernel():
    # eq. (11.3): K(s - t) = integral e^{-i <s-t, lambda>} dmu(lambda).
    # The numerical spectral integral must reproduce the closed form.
    res = ghosal_statgp_spec(h=0.6, n_grid=4000, lam_max=30.0)
    assert res["bochner_gap"] < 1e-9
    assert res["estimate"] == pytest.approx(res["kernel_exact"], rel=1e-9)


def test_the_kernel_is_one_at_zero_separation_and_decays():
    near = ghosal_statgp_spec(h=0.0, n_grid=2000, lam_max=30.0)["kernel_exact"]
    far = ghosal_statgp_spec(h=2.0, n_grid=2000, lam_max=30.0)["kernel_exact"]
    assert near == pytest.approx(1.0, rel=1e-9)
    assert far < near
