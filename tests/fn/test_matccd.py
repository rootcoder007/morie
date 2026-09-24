"""Tests for matccd.matched_case_control."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.matccd import matched_case_control


def test_matccd_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    # 20 matched sets of size 2 (one case, one control)
    matching_id = [i // 2 for i in range(n)]
    # exactly one case per set
    cases = [1 if i % 2 == 0 else 0 for i in range(n)]
    controls = [1 - c for c in cases]
    exposure = rng.normal(0, 1, n)

    result = matched_case_control(cases, controls, matching_id, exposure)

    # result is a dict-like RichResult; check all advertised keys are present
    expected_keys = (
        "estimate", "log_or", "se", "ci", "information",
        "loglik", "n_sets", "n_obs", "iters", "converged",
    )
    for key in expected_keys:
        assert key in result

    # derived from the input
    assert result["n_obs"] == n
    assert result["n_sets"] == 20
    # odds ratio should be a finite real number
    assert math.isfinite(result["estimate"])


def test_matccd_edge():
    """Test edge cases."""
    # Empty input is invalid per docstring ("no observations")
    with pytest.raises(ValueError):
        matched_case_control([], [], [], [])
