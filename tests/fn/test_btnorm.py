"""Tests for btnorm.boot_normal_ci."""

import math

from morie.fn import _array_core as np

from morie.fn.btnorm import boot_normal_ci


def test_btnorm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    theta_hat = 1.5
    theta_b = rng.normal(1.5, 1, 200)
    alpha = 0.05
    result = boot_normal_ci(theta_hat, theta_b, alpha)
    assert isinstance(result, dict)
    payload = result["payload"] if "payload" in result else result
    assert "lo" in payload
    assert "hi" in payload
    assert "bias" in payload
    assert "se_b" in payload
    assert "skew" in payload
    assert math.isfinite(payload["lo"])
    assert math.isfinite(payload["hi"])
    assert math.isfinite(payload["se_b"])
    assert payload["lo"] <= payload["hi"]


def test_btnorm_edge():
    """Test edge cases with a small but valid sample of replicates."""
    rng = np.random.default_rng(42)
    theta_hat = 0.0
    theta_b = rng.normal(0.0, 1, 5)
    alpha = 0.05
    result = boot_normal_ci(theta_hat, theta_b, alpha)
    assert isinstance(result, dict)
    payload = result["payload"] if "payload" in result else result
    assert payload["B"] == 5
    assert math.isfinite(payload["lo"])
    assert math.isfinite(payload["hi"])
    assert math.isfinite(payload["se_b"])
    assert payload["lo"] <= payload["hi"]
