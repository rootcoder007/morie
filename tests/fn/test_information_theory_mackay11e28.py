"""Verification tests for information_theory_mackay11e28.gchpost.

The expected values are recomputed from MacKay (2003) eq. (11.27)-(11.29) p. 182 in the test body, so a
drift in the implementation fails the test.
"""

import math

import pytest

from morie.fn.information_theory_mackay11e28 import gchpost


def test_gchpost_is_the_gaussian_conjugate_posterior():
    y, v, s2 = 1.7, 2.0, 0.5
    var = 1.0 / (1.0 / v + 1.0 / s2)
    res = gchpost(y, v, s2)
    assert res["var"] == pytest.approx(var, rel=1e-12)
    assert res["mean"] == pytest.approx(v / (v + s2) * y, rel=1e-12)
    assert res["sd"] == pytest.approx(math.sqrt(var), rel=1e-12)
    assert res["marginalvar"] == pytest.approx(v + s2, rel=1e-12)


def test_gchpost_shrinks_towards_the_prior_as_the_channel_gets_noisier():
    quiet = gchpost(1.0, 1.0, 0.01)
    noisy = gchpost(1.0, 1.0, 100.0)
    assert quiet["mean"] > 0.9
    assert noisy["mean"] < 0.02


def test_gchpost_requires_positive_variances():
    with pytest.raises(ValueError):
        gchpost(1.0, 0.0, 1.0)
