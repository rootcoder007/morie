"""Tests for mlpv.multilevel_pseudo_variance_ratio."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.mlpv import multilevel_pseudo_variance_ratio


def test_mlpv_basic():
    """Test basic functionality with reasonable inputs."""
    rng_y = np.random.default_rng(43)
    rng_X = np.random.default_rng(42)
    rng_c = np.random.default_rng(41)

    n = 100
    p = 5
    J = 5

    y = rng_y.normal(0, 1, n)
    X = rng_X.normal(0, 1, (n, p))
    cluster = rng_c.integers(0, J, n)

    result = multilevel_pseudo_variance_ratio(y, X, cluster)

    # The function returns a dict-like RichResult
    assert isinstance(result, dict)

    # Check that all documented keys are present
    expected_keys = {
        "estimate",
        "pr",
        "sigma2_null",
        "sigma2_full",
        "df_null",
        "df_full",
        "n_clusters",
        "p",
        "n",
    }
    assert expected_keys.issubset(result.keys())

    # Basic sanity checks on numeric results
    assert math.isfinite(result["estimate"])
    assert result["sigma2_null"] >= 0
    assert result["sigma2_full"] >= 0

    # Metadata checks
    assert result["n"] == n
    assert result["p"] == p
    assert result["n_clusters"] == J
    assert result["df_null"] > 0
    assert result["df_full"] > 0


def test_mlpv_edge():
    """Test edge cases: minimal valid input and invalid configurations."""
    # Minimal valid configuration: n=10, p=1, J=2 -> n - J - p = 7 > 0
    rng_y = np.random.default_rng(7)
    rng_X = np.random.default_rng(8)
    rng_c = np.random.default_rng(9)

    n = 10
    p = 1
    J = 2

    y = rng_y.normal(0, 1, n)
    X = rng_X.normal(0, 1, (n, p))
    cluster = rng_c.integers(0, J, n)

    result = multilevel_pseudo_variance_ratio(y, X, cluster)

    assert isinstance(result, dict)
    assert math.isfinite(result["estimate"])
    assert result["n"] == n
    assert result["p"] == p
    assert result["n_clusters"] == J

    # Invalid input: mismatched lengths should raise ValueError
    with pytest.raises(ValueError):
        multilevel_pseudo_variance_ratio(y, X, cluster[:-1])

    # Invalid input: too many predictors for the data (n - J - p <= 0)
    # According to the docstring this should raise ValueError.
    with pytest.raises(ValueError):
        multilevel_pseudo_variance_ratio(
            y,
            rng_X.normal(0, 1, (n, n - 1)),  # p = n-1 = 9
            rng_c.integers(0, 1, n),  # single cluster (J = 1)
        )
