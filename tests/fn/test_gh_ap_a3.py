"""Tests for gh_ap_a3.ghosal_tv_distance."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_a3 import ghosal_tv_distance


def test_gh_ap_a3_basic():
    """Test basic functionality: identical distributions yield zero TV distance."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_tv_distance(x, x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    # When p == q, both forms of the formula must give zero.
    assert abs(float(result["estimate"])) < 1e-12
    assert "sup_form" in result
    assert "forms_agree" in result
    assert result["forms_agree"] is True or result["forms_agree"] == True


def test_gh_ap_a3_known_value():
    """Test against an independently computed total variation distance.

    For p = [3, 1] and q = [1, 3], the formula gives
    (1/2) * ||p_normalized - q_normalized||_1, where the L1 difference
    is computed from the normalized weights.
    """
    p = np.array([3.0, 1.0])
    q = np.array([1.0, 3.0])

    p_sum = float(np.asarray(p, dtype=float).sum())
    q_sum = float(np.asarray(q, dtype=float).sum())
    p_norm = [float(v) / p_sum for v in np.asarray(p, dtype=float)]
    q_norm = [float(v) / q_sum for v in np.asarray(q, dtype=float)]
    l1 = sum(abs(a - b) for a, b in zip(p_norm, q_norm))
    expected = 0.5 * l1

    result = ghosal_tv_distance(p, q)
    assert "estimate" in result
    assert abs(float(result["estimate"]) - expected) < 1e-12
    # The sup_A form equals the half-L1 form for these inputs as well.
    assert "sup_form" in result
    assert abs(float(result["sup_form"]) - expected) < 1e-12
    assert result["forms_agree"] is True or result["forms_agree"] == True


def test_gh_ap_a3_edge():
    """Test edge case: a single-element distribution against itself."""
    result = ghosal_tv_distance(np.array([42.0]), np.array([42.0]))
    assert "estimate" in result
    # Identical one-element distributions => zero TV distance.
    assert abs(float(result["estimate"])) < 1e-12
