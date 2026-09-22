"""Tests for gh_c13_16.ghosal_bb_censored."""

from morie.fn import _array_core as np

from morie.fn.gh_c13_16 import ghosal_bb_censored


def test_gh_c13_16_basic():
    """Test basic functionality: KM limit of the censored Bayesian bootstrap."""
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    events = np.array([1, 1, 0, 1, 0])
    t_query = 4.0

    result = ghosal_bb_censored(times, events, t_query)

    # Documented keys
    assert "estimate" in result
    assert "km_survival" in result
    assert "method" in result

    # Independent computation of KM estimate using plain arithmetic.
    # Sort by time; at each event time t with at_risk R and deaths d,
    # survival factor = (R - d) / R. Censored observations do not
    # trigger a factor but reduce the risk set.
    order = sorted(range(len(times)), key=lambda i: times[i])
    t_sorted = [times[i] for i in order]
    e_sorted = [events[i] for i in order]
    at_risk = len(t_sorted)
    surv = 1.0
    for i in range(len(t_sorted)):
        if t_sorted[i] > t_query:
            break
        if e_sorted[i] > 0:
            surv *= (at_risk - 1.0) / at_risk
        at_risk -= 1

    est = float(np.asarray(result["estimate"], dtype=float))
    km = float(np.asarray(result["km_survival"], dtype=float))

    assert np.isfinite(est)
    assert np.isclose(est, surv)
    assert np.isclose(km, surv)
    # Both documented keys should agree at this limit.
    assert np.isclose(est, km)

    # Sanity: probability in [0, 1].
    assert 0.0 <= est <= 1.0
    assert 0.0 <= km <= 1.0


def test_gh_c13_16_edge():
    """Test edge case: single observation, query after the event."""
    times = np.array([42.0])
    events = np.array([1])
    t_query = 100.0

    result = ghosal_bb_censored(times, events, t_query)

    # Single event: survival drops by factor (1 - 1/1) = 0 at t_query >= 42.
    assert np.isclose(float(result["estimate"]), 0.0)
    assert np.isclose(float(result["km_survival"]), 0.0)
    assert "method" in result


def test_gh_c13_16_no_event_before_query():
    """If no events occur at or before t_query, survival stays at 1."""
    times = np.array([1.0, 2.0, 3.0])
    events = np.array([0, 0, 0])
    t_query = 2.5

    result = ghosal_bb_censored(times, events, t_query)

    assert np.isclose(float(result["estimate"]), 1.0)
    assert np.isclose(float(result["km_survival"]), 1.0)
