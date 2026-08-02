"""grdcl: gradient clipping by global norm (Pascanu et al. 2013).

    g <- g * min(1, max_norm / ||g||)
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.grdcl import gradient_clipping as clip


def test_grdcl_short_gradient_is_left_exactly_alone():
    """Below the threshold the coefficient is 1 and nothing changes -- this
    is what makes clipping safe to leave switched on."""
    g = np.array([0.3, 0.4])          # norm 0.5
    r = clip(g, max_norm=1.0)
    assert r["total_norm"] == pytest.approx(0.5)
    assert r["clip_coef"] == pytest.approx(1.0)
    assert np.asarray(r["tensor"]) == pytest.approx(g)


def test_grdcl_long_gradient_is_rescaled_to_exactly_max_norm():
    g = np.array([3.0, 4.0])          # norm 5
    out = np.asarray(clip(g, max_norm=1.0)["tensor"])
    assert np.linalg.norm(out) == pytest.approx(1.0)
    assert out == pytest.approx(g / 5.0)


def test_grdcl_preserves_direction():
    """Clipping is a pure rescale: the unit vector must be unchanged."""
    rng = np.random.default_rng(1637)
    g = rng.standard_normal(20) * 50
    out = np.asarray(clip(g, max_norm=2.0)["tensor"])
    assert out / np.linalg.norm(out) == pytest.approx(g / np.linalg.norm(g))


def test_grdcl_is_idempotent():
    """Clipping an already-clipped gradient changes nothing."""
    rng = np.random.default_rng(1643)
    once = np.asarray(clip(rng.standard_normal(15) * 30, max_norm=1.5)["tensor"])
    twice = np.asarray(clip(once, max_norm=1.5)["tensor"])
    assert twice == pytest.approx(once)


def test_grdcl_uses_the_GLOBAL_norm_not_per_element():
    """Two coordinates each below max_norm can still exceed it jointly.

    (0.8, 0.8) has no element above 1 but a norm of 1.13, so it must clip --
    that is the difference between global-norm and per-element clipping.
    """
    r = clip(np.array([0.8, 0.8]), max_norm=1.0)
    assert r["total_norm"] == pytest.approx(np.sqrt(1.28))
    assert r["clip_coef"] < 1.0
    assert np.linalg.norm(np.asarray(r["tensor"])) == pytest.approx(1.0)


def test_grdcl_zero_gradient_does_not_divide_by_zero():
    r = clip(np.zeros(5), max_norm=1.0)
    assert np.all(np.isfinite(np.asarray(r["tensor"])))
    assert r["total_norm"] == pytest.approx(0.0)
