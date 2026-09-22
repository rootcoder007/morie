"""Tests for eslo63.esl_oob_632."""

from morie.fn import _array_core as np

from morie.fn.eslo63 import esl_oob_632


def test_eslo63_basic():
    """Test basic .632 estimator with documented scalar inputs."""
    err_train = 0.10
    err_boot = 0.20
    result = esl_oob_632(err_train, err_boot)
    assert isinstance(result, dict)
    assert "value" in result
    assert "err_632" in result
    assert "err_632_plus" in result
    assert result["err_632_plus"] is None  # no gamma supplied
    expected = 0.368 * err_train + 0.632 * err_boot
    assert abs(result["value"] - expected) < 1e-12
    assert abs(result["err_632"] - expected) < 1e-12


def test_eslo63_edge():
    """Test .632+ with gamma supplied via (p1, q1) formula (7.59)."""
    err_train = 0.0
    err_boot = 0.5
    # dichotomous no-information rate from (7.59): p1*(1-q1) + (1-p1)*q1
    p1, q1 = 0.5, 0.5
    gamma = p1 * (1 - q1) + (1 - p1) * q1
    result = esl_oob_632(err_train, err_boot, p1=p1, q1=q1)
    assert isinstance(result, dict)
    assert result["err_632_plus"] is not None
    # 1-NN counterexample: R = 1, w = .632 / (1 - .368) = 1
    expected_w = 0.632 / (1.0 - 0.368)
    expected_632p = (1.0 - expected_w) * err_train + expected_w * err_boot
    assert abs(result["weight"] - expected_w) < 1e-12
    assert abs(result["err_632_plus"] - expected_632p) < 1e-12
    assert abs(result["value"] - 0.632 * 0.5) < 1e-12
