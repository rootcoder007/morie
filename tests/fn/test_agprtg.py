"""Tests for agprtg.alphazero_priority_target."""

from morie.fn import _array_core as np

from morie.fn.agprtg import alphazero_priority_target


def test_agprtg_basic():
    """Test basic functionality."""
    priorities = np.random.default_rng(42).uniform(0.0, 1.0, 100)
    z = np.random.default_rng(7).normal(0, 1, 100)
    v = np.random.default_rng(13).normal(0, 1, 100)
    result = alphazero_priority_target(
        replay_buffer=None, priorities=priorities, z=z, v=v,
        alpha=0.6, beta=0.4, eps=1e-6, variant="proportional",
    )
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "prob" in result
    assert "weight" in result
    assert "priority" in result

    # Independent computation of the formula.
    raw_p = [float(x) + 1e-6 for x in priorities]
    pa = [float(x) ** 0.6 for x in raw_p]
    tot = 0.0
    for x in pa:
        tot += x
    prob_expected = [x / tot if tot > 0.0 else 0.0 for x in pa]
    n = len(prob_expected)
    w_expected = [(n * q) ** (-0.4) if q > 0.0 else 0.0 for q in prob_expected]
    mx = 0.0
    for x in w_expected:
        if x > mx:
            mx = x
    w_expected = [x / mx if mx > 0.0 else 0.0 for x in w_expected]

    assert result["estimate"] == prob_expected[0]
    assert len(result["prob"]) == n
    assert len(result["weight"]) == n
    assert len(result["priority"]) == n
    # Sums match the expected probability distribution.
    prob_sum = 0.0
    for x in result["prob"]:
        prob_sum += x
    assert abs(prob_sum - 1.0) < 1e-9
    # The max importance weight should be 1.0 by construction.
    w_max = 0.0
    for x in result["weight"]:
        if x > w_max:
            w_max = x
    assert abs(w_max - 1.0) < 1e-12


def test_agprtg_edge():
    """Test edge cases."""
    priorities = np.random.default_rng(42).uniform(0.0, 1.0, 50)
    z = np.random.default_rng(7).normal(0, 1, 50)
    v = np.random.default_rng(13).normal(0, 1, 50)
    result = alphazero_priority_target(
        replay_buffer=None, priorities=priorities, z=z, v=v,
        alpha=0.6, beta=0.4, eps=1e-6, variant="proportional",
    )
    assert isinstance(result, dict)
    assert "estimate" in result
    assert len(result["prob"]) == 50
