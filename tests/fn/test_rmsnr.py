"""rmsnr: root-mean-square normalisation (Zhang & Sennrich 2019).

    y = gamma * x / sqrt(mean(x^2) + eps)

RMSNorm differs from LayerNorm precisely by NOT subtracting the mean.
"""

import numpy as np
import pytest

from morie.fn.rmsnr import rms_norm as rn


def test_rmsnr_matches_the_closed_form():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    eps = 1e-8
    rms = np.sqrt((x**2).mean(axis=-1, keepdims=True) + eps)
    assert np.asarray(rn(x, eps=eps)["tensor"]) == pytest.approx(x / rms)


def test_rmsnr_output_has_unit_root_mean_square():
    rng = np.random.default_rng(1621)
    y = np.asarray(rn(rng.normal(0, 5, (8, 32)))["tensor"])
    assert np.sqrt((y**2).mean(axis=-1)) == pytest.approx(np.ones(8), abs=1e-5)


def test_rmsnr_does_not_centre_the_input():
    """The distinguishing property vs LayerNorm: a non-zero mean survives.

    An all-positive row stays all-positive; LayerNorm would straddle zero.
    """
    y = np.asarray(rn(np.array([[1.0, 2.0, 3.0, 4.0]]))["tensor"])
    assert np.all(y > 0)
    assert y.mean() > 0.5


def test_rmsnr_is_scale_invariant_up_to_eps():
    """Multiplying the input by c leaves the output essentially unchanged.

    Only ESSENTIALLY: the eps floor breaks exact invariance, because

        cx / sqrt(c^2 mean(x^2) + eps) = x / sqrt(mean(x^2) + eps/c^2)

    so scaling up shrinks the effective eps. At c = 13 with the default eps
    the gap is ~1.3e-06 -- small, real, and not a defect. Driving eps to zero
    recovers exact invariance, which is what the second half checks.
    """
    rng = np.random.default_rng(1627)
    x = rng.standard_normal((4, 16))
    assert np.asarray(rn(x * 13.0)["tensor"]) == pytest.approx(
        np.asarray(rn(x)["tensor"]), abs=1e-5
    )
    # With a negligible eps the invariance is exact to floating point.
    tiny = 1e-300
    assert np.asarray(rn(x * 13.0, eps=tiny)["tensor"]) == pytest.approx(
        np.asarray(rn(x, eps=tiny)["tensor"]), rel=1e-12
    )


def test_rmsnr_gamma_scales_each_feature():
    rng = np.random.default_rng(1631)
    x = rng.standard_normal((3, 5))
    g = np.array([1.0, 2.0, 3.0, 0.5, -1.0])
    assert np.asarray(rn(x, gamma=g)["tensor"]) == pytest.approx(
        np.asarray(rn(x)["tensor"]) * g
    )


def test_rmsnr_all_zero_row_stays_finite():
    y = np.asarray(rn(np.zeros((1, 6)))["tensor"])
    assert np.all(np.isfinite(y))
    assert np.allclose(y, 0.0)
