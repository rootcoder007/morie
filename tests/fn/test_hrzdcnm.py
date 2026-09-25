"""Tests for hrzdcnm.horowitz_deconv_normality (Horowitz sec. 5.1.3)."""

import math

import pytest

from morie.fn.hrzdcnm import horowitz_deconv_normality


def test_hrzdcnm_basic():
    """z = [n h / b]^{1/2} (f_n - f - bias) / sigma and the two-sided
    normal p-value 2 (1 - Phi(|z|)) = erfc(|z| / sqrt 2)."""
    r = horowitz_deconv_normality(0.43, 0.40, 500, 0.3, 2.0, bias=0.01, sigma=0.8)
    sc = math.sqrt(500 * 0.3 / 2.0)
    z = sc * (0.43 - 0.40 - 0.01) / 0.8
    assert r["scaling"] == pytest.approx(sc, rel=1e-15)
    assert r["z"] == pytest.approx(z, rel=1e-12)
    assert r["p_two_sided"] == pytest.approx(math.erfc(abs(z) / math.sqrt(2)), rel=1e-12)
    assert r["bias_subtracted"] == 0.01


def test_hrzdcnm_edge():
    """Non-positive h, b or sigma, or n < 2, raise."""
    for args in ((0.4, 0.4, 1, 0.3, 2.0), (0.4, 0.4, 50, 0.0, 2.0), (0.4, 0.4, 50, 0.3, -1.0)):
        with pytest.raises(ValueError):
            horowitz_deconv_normality(*args)
    with pytest.raises(ValueError):
        horowitz_deconv_normality(0.4, 0.4, 50, 0.3, 2.0, sigma=0.0)
