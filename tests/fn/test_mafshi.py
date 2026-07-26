"""mafshi: inverse Fisher z (Fisher 1921)."""

import numpy as np
import pytest

from morie.fn.mafshi import ma_fishers_z_inverse as z_to_r


def test_mafshi_matches_the_printed_closed_form():
    """r = (e^2z - 1)/(e^2z + 1), evaluated independently of tanh."""
    for z in (-3.0, -0.5, 0.0, 0.25, 1.0, 5.0):
        e = np.exp(2.0 * z)
        assert z_to_r(z)["r"] == pytest.approx((e - 1) / (e + 1), rel=1e-14)


def test_mafshi_known_values():
    assert z_to_r(0.0)["r"] == 0.0
    assert z_to_r(0.5493061443340548)["r"] == pytest.approx(0.5, rel=1e-13)


def test_mafshi_saturates_instead_of_overflowing():
    """At large z the literal (e^2z - 1)/(e^2z + 1) is inf/inf = nan.

    tanh returns 1.0. This is the reason the implementation does not follow
    the printed expression literally, and the reason it is worth a test.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(2.0 * 400.0)
        assert np.isinf(e) and np.isnan((e - 1) / (e + 1))
    assert z_to_r(400.0)["r"] == 1.0
    assert z_to_r(-400.0)["r"] == -1.0


def test_mafshi_stays_inside_the_correlation_range():
    rng = np.random.default_rng(67)
    out = np.asarray(z_to_r(rng.uniform(-20, 20, 500))["r"])
    assert np.all(np.abs(out) <= 1.0)


def test_mafshi_is_odd_and_monotone():
    assert z_to_r(1.3)["r"] == pytest.approx(-z_to_r(-1.3)["r"])
    rs = [z_to_r(z)["r"] for z in np.linspace(-4, 4, 40)]
    assert rs == sorted(rs)


def test_mafshi_back_transform_is_not_the_mean():
    """tanh is concave for z > 0, so mean(tanh(z)) != tanh(mean(z)).

    Pooling on the r scale and pooling on the z scale genuinely differ; this
    pins that the function does not quietly average.
    """
    zs = np.array([0.1, 2.0])
    assert z_to_r(float(zs.mean()))["r"] != pytest.approx(
        float(np.mean(np.asarray(z_to_r(zs)["r"]))), abs=1e-6
    )


def test_mafshi_rejects_non_finite():
    with pytest.raises(ValueError, match="finite"):
        z_to_r(np.inf)
