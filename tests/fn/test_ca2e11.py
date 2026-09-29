"""Tests for ca2e11.ca_chapter_2_equation_11."""

from morie.fn import _array_core as np
from morie.fn.ca2e11 import ca_chapter_2_equation_11


def test_ca2e11_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0, 1, 100)
    yhat = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_equation_11(y, yhat)
    assert isinstance(result, dict)
    assert "var_total" in result


def test_ca2e11_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0, 1, 100)
    yhat = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_equation_11(y, yhat)
    assert isinstance(result, dict)


def test_variance_partition_recomputed():
    """Weisburd et al. (2022) eqs 2.11-2.14 with divisor n."""
    import pytest

    y = [3.0, 5.0, 4.0, 8.0, 6.0, 2.0]
    yhat = [3.5, 4.5, 4.2, 7.1, 6.3, 2.4]
    n = 6
    yb = sum(y) / n
    sst = sum((v - yb) ** 2 for v in y)
    ssm = sum((v - yb) ** 2 for v in yhat)
    ssr = sum((a - b) ** 2 for a, b in zip(y, yhat))
    want = {11: sst / n, 12: ssm / n, 13: ssr / n, 14: ssm / sst}[11]
    r = ca_chapter_2_equation_11(y, yhat)
    assert r["value"] == pytest.approx(want, rel=1e-14)
    assert (r["var_total"], r["var_model"], r["var_resid"], r["r2"]) == pytest.approx(
        (sst / n, ssm / n, ssr / n, ssm / sst), rel=1e-14
    )
