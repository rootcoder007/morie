"""Tests for midor.model_identify_estimate_refute."""

import math

from morie.fn import _array_core as np

from morie.fn.midor import model_identify_estimate_refute


def test_midor_basic():
    """Test basic functionality on a 3-node DAG."""
    rng = np.random.default_rng(42)
    # 3-node DAG: node 0 -> node 1 -> node 2 and 0 -> 2
    dag = [
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
    ]
    n, p = 40, 3
    data = rng.normal(0, 1, (n, p))
    result = model_identify_estimate_refute(
        dag, data, treatment=0, outcome=2, seed=42
    )
    assert isinstance(result, dict)
    expected = {
        "estimate", "se", "ci_lower", "ci_upper",
        "identified", "adjustment_set", "all_backdoor_sets",
        "placebo_effect", "random_cause_effect",
        "subset_sd", "refutations_passed",
    }
    assert expected.issubset(set(result.keys()))
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["se"])
    assert math.isfinite(result["ci_lower"])
    assert math.isfinite(result["ci_upper"])
    assert result["ci_lower"] <= result["ci_upper"]
    assert isinstance(result["identified"], bool)


def test_midor_edge():
    """Test edge case with a minimal 2-node DAG."""
    rng = np.random.default_rng(0)
    # 2-node DAG: 0 -> 1
    dag = [
        [0, 1],
        [0, 0],
    ]
    n, p = 20, 2
    data = rng.normal(0, 1, (n, p))
    result = model_identify_estimate_refute(
        dag, data, treatment=0, outcome=1, seed=0
    )
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "se" in result
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["se"])
