"""Tests for dpoF.dpo_loss."""

import math

from morie.fn import _array_core as np
from morie.fn.dpoF import dpo_loss


def test_dpoF_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    logp_w = [-abs(x) for x in rng.normal(0, 1, n)]
    logp_l = [-abs(x) for x in rng.normal(0, 1, n)]
    logp_ref_w = [-abs(x) for x in rng.normal(0, 1, n)]
    logp_ref_l = [-abs(x) for x in rng.normal(0, 1, n)]
    beta = 0.8
    result = dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, beta)
    assert isinstance(result, dict)
    has_finite = False
    for v in result.values():
        if isinstance(v, (int, float)) and math.isfinite(v):
            has_finite = True
            break
    assert has_finite


def test_dpoF_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 20
    logp_w = [-abs(x) for x in rng.normal(0, 1, n)]
    logp_l = [-abs(x) for x in rng.normal(0, 1, n)]
    logp_ref_w = [-abs(x) for x in rng.normal(0, 1, n)]
    logp_ref_l = [-abs(x) for x in rng.normal(0, 1, n)]
    beta = 0.5
    result = dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, beta)
    assert isinstance(result, dict)
    has_finite = False
    for v in result.values():
        if isinstance(v, (int, float)) and math.isfinite(v):
            has_finite = True
            break
    assert has_finite
