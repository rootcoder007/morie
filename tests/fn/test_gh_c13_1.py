"""Tests for gh_c13_1.ghosal_surv_dp_post."""

from morie.fn import _array_core as np

from morie.fn.gh_c13_1 import ghosal_surv_dp_post


def test_gh_c13_1_basic():
    """Test basic functionality."""
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    events = np.array([1, 1, 1, 1, 1])
    t_query = 3.0
    result = ghosal_surv_dp_post(times, events, t_query)
    assert "estimate" in result
    assert "survival_at_t" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))

    # Independent computation of the formula:
    # S(t) = prod_{t_j <= t_query, event=1}
    #          (alpha * exp(-t_j) + R_j - 1) / (alpha * exp(-t_j) + R_j)
    # with R_j = number-at-risk at distinct event time t_j (in this test
    # all times are distinct), alpha = 2.0.
    alpha = 2.0
    order = sorted(range(len(times)), key=lambda i: times[i])
    at_risk = len(times)
    expected_surv = 1.0
    for i in order:
        if times[i] > t_query:
            break
        if events[i] > 0:
            S0 = np.exp(-times[i])
            expected_surv *= (alpha * S0 + at_risk - 1.0) \
                / (alpha * S0 + at_risk)
        at_risk -= 1
    assert np.isclose(float(result["estimate"]), float(expected_surv))
    assert np.isclose(float(result["survival_at_t"]), float(expected_surv))


def test_gh_c13_1_edge():
    """Test edge cases."""
    times = np.array([42.0])
    events = np.array([1])
    result = ghosal_surv_dp_post(times, events, 42.0)
    # t_query == event time -> event contributes once, with R=1.
    alpha = 2.0
    S0 = np.exp(-42.0)
    expected = (alpha * S0 + 1.0 - 1.0) / (alpha * S0 + 1.0)
    assert np.isclose(float(result["estimate"]), float(expected))
