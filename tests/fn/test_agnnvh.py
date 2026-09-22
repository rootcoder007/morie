"""Tests for agnnvh.alphazero_value_head."""

import math

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.agnnvh import alphazero_value_head


def test_agnnvh_basic():
    """Test basic functionality against the documented AlphaZero loss formula."""
    z = 1.0
    v = 0.5
    # pi must be a normalised policy (sums to 1); p must be a same-length policy.
    pi = [0.5, 0.25, 0.25]
    p = [0.4, 0.4, 0.2]
    theta = [0.1, -0.2, 0.3, 0.0]
    c = 1e-4

    result = alphazero_value_head(z, v, pi, p, theta, c)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "value_loss" in result
    assert "policy_loss" in result
    assert "l2" in result
    assert "sq_norm" in result

    # Independent recomputation from the docstring formula.
    zz = float(z)
    vv = float(v)
    exp_vloss = (zz - vv) ** 2
    exp_ploss = 0.0
    for i in range(len(pi)):
        if pi[i] > 0.0:
            qq_i = p[i] if p[i] > 0.0 else 1e-12
            exp_ploss -= pi[i] * math.log(qq_i)
    exp_sq = 0.0
    for x in theta:
        exp_sq += x * x
    exp_l2 = float(c) * exp_sq
    exp_total = exp_vloss + exp_ploss + exp_l2

    assert math.isclose(result["value_loss"], exp_vloss, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["policy_loss"], exp_ploss, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["l2"], exp_l2, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["sq_norm"], exp_sq, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["estimate"], exp_total, rel_tol=1e-12, abs_tol=1e-12)
    # The total loss is the sum of its three named parts.
    assert math.isclose(
        result["estimate"],
        result["value_loss"] + result["policy_loss"] + result["l2"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_agnnvh_edge():
    """Test edge cases with theta=None and trivial policy."""
    z = -1.0
    v = 0.0
    pi = [1.0]
    p = [1.0]
    theta = None
    c = 1e-4

    result = alphazero_value_head(z, v, pi, p, theta, c)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "value_loss" in result
    assert "policy_loss" in result
    assert "l2" in result
    assert "sq_norm" in result

    # With theta=None the L2 term and squared norm must both be 0.
    assert result["l2"] == 0.0
    assert result["sq_norm"] == 0.0

    # Value loss is (z - v)^2 = 1.0 here.
    assert math.isclose(result["value_loss"], 1.0, rel_tol=1e-12, abs_tol=1e-12)
    # Policy loss: -1 * log(1) = 0 when pi and p both put mass on the same action.
    assert math.isclose(result["policy_loss"], 0.0, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["estimate"], 1.0, rel_tol=1e-12, abs_tol=1e-12)
