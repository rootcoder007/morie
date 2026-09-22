"""Tests for fzt51.fauzi_thm5_1_naive_kernel_equiv."""

from morie.fn import _array_core as np

from morie.fn.fzt51 import fauzi_thm5_1_naive_kernel_equiv


def test_fzt51_basic():
    """Test basic functionality with documented scalar arguments."""
    ks_emp = 0.12
    ks_kernel = 0.10
    cvm_emp = 0.25
    cvm_kernel = 0.23

    result = fauzi_thm5_1_naive_kernel_equiv(ks_emp, ks_kernel, cvm_emp, cvm_kernel)

    assert isinstance(result, dict)

    # Check that the documented keys are present.
    for key in ("ksdiff", "cvmdiff", "close", "tol", "method"):
        assert key in result

    # Compute expected differences independently from the formula.
    expected_ksdiff = abs(float(ks_emp) - float(ks_kernel))
    expected_cvmdiff = abs(float(cvm_emp) - float(cvm_kernel))

    assert result["ksdiff"] == expected_ksdiff
    assert result["cvmdiff"] == expected_cvmdiff

    # Default tolerance is 0.05; both differences should be below it.
    assert result["tol"] == 0.05
    assert result["close"] is True
    assert result["method"] == "naive kernel vs empirical GOF equivalence (Theorem 5.1)"


def test_fzt51_edge():
    """Test edge case where differences exceed the tolerance."""
    ks_emp = 0.30
    ks_kernel = 0.10
    cvm_emp = 0.40
    cvm_kernel = 0.20
    tol = 0.05

    result = fauzi_thm5_1_naive_kernel_equiv(
        ks_emp, ks_kernel, cvm_emp, cvm_kernel, tol=tol
    )

    assert isinstance(result, dict)

    expected_ksdiff = abs(float(ks_emp) - float(ks_kernel))
    expected_cvmdiff = abs(float(cvm_emp) - float(cvm_kernel))

    assert result["ksdiff"] == expected_ksdiff
    assert result["cvmdiff"] == expected_cvmdiff
    assert result["tol"] == tol
    # Both differences exceed tolerance, so not close.
    assert result["close"] is False
