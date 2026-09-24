"""Tests for itrct1.interaction_did."""

import math

from morie.fn import _array_core as np

from morie.fn.itrct1 import interaction_did


def test_itrct1_basic():
    """Test basic functionality with balanced assignment and covariates."""
    rng_y = np.random.default_rng(43)
    rng_x = np.random.default_rng(40)
    n = 60
    # 3 strata; alternating D guarantees both arms within every level of V
    V = [float(i % 3) for i in range(n)]
    D = [float(i % 2) for i in range(n)]
    y = rng_y.normal(0, 1, n)
    X = rng_x.normal(0, 1, (n, 3))

    result = interaction_did(y, D, V, X)
    assert isinstance(result, dict)
    # Real keys exposed by the returned RichResult payload
    assert "estimate" in result
    assert "se" in result
    assert "att" in result
    assert "att_se" in result
    assert "levels" in result
    assert "level_n" in result
    assert "n_levels" in result
    assert "n" in result
    assert "method" in result
    assert result["n_levels"] == 3
    assert result["n"] == n
    assert len(result["att"]) == 3
    assert result["levels"] == [0.0, 1.0, 2.0]
    assert result["level_n"] == [20, 20, 20]
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["se"])
    for v in result["att"]:
        assert math.isfinite(v)
    for v in result["att_se"]:
        assert math.isfinite(v)


def test_itrct1_edge():
    """Test edge case: two levels with no covariates (X=None)."""
    rng_y = np.random.default_rng(43)
    n = 40
    V = [0.0] * 20 + [1.0] * 20
    D = [float(i % 2) for i in range(n)]
    y = rng_y.normal(0, 1, n)

    result = interaction_did(y, D, V, None)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "se" in result
    assert "att" in result
    assert "levels" in result
    assert "n_levels" in result
    assert "n" in result
    assert result["n_levels"] == 2
    assert result["n"] == n
    assert result["level_n"] == [20, 20]
    assert result["levels"] == [0.0, 1.0]
    assert len(result["att"]) == 2
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["se"])
