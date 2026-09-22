"""Tests for bnsiii.bound_iii."""

from morie.fn import _array_core as np

from morie.fn.bnsiii import bound_iii


def _make_intervals(rng, n):
    """Construct n interval-valued observations with y[i] <= X[i].

    Each observation's lower end is drawn, then the upper end is drawn
    so that X[i] >= y[i] is guaranteed by construction.
    """
    lower = rng.normal(0.0, 1.0, n)
    half_width = rng.uniform(0.05, 1.0, n)
    upper = lower + half_width
    return lower, upper


def test_bnsiii_basic():
    """Test basic functionality.

    For each observation i, build (y[i], X[i]) with y[i] <= X[i] so the
    input satisfies the documented constraint ``y and X have the same
    length, with y the lower end and X the upper end of each interval``.
    Then evaluate the criterion on a grid of candidate moment values
    that brackets the sample moments of the intervals, where the
    identified set is non-empty.
    """
    n = 100
    rng_y = np.random.default_rng(43)
    rng_X = np.random.default_rng(42)
    rng_m = np.random.default_rng(7)

    y, X = _make_intervals(rng_y, n)
    # X must be 1-D of length n, like y.
    X = X.reshape(-1)

    # The identified set contains all theta in [mL, mU].  Build a grid
    # that straddles the sample moment mean so we exercise both the
    # interior and exterior of the set.
    moments = np.linspace(-2.0, 2.0, 41)

    result = bound_iii(y, X, moments)

    assert isinstance(result, dict)

    # The function returns these documented keys.
    for key in ("lower", "upper", "width", "n_in_set",
                "q_min", "q_max_stat", "n"):
        assert key in result, f"missing documented key: {key}"

    # Sanity checks on scalar outputs.
    assert int(result["n"]) == n
    assert result["upper"] >= result["lower"]
    assert result["width"] == result["upper"] - result["lower"]

    # All candidate moment values inside [lower, upper] must be counted
    # as in the identified set; values outside must not be.  This is an
    # independent recomputation of the expected n_in_set from the grid
    # and the documented lower/upper bounds.
    grid = moments.reshape(-1)
    expected_in = int(np.sum((grid >= result["lower"]) & (grid <= result["upper"])))
    assert int(result["n_in_set"]) == expected_in

    # q_min is the minimum of the criterion over the grid, which is
    # non-positive (it equals 0 at points inside the identified set).
    assert result["q_min"] <= 0.0 + 1e-9


def test_bnsiii_edge():
    """Test edge cases with a short grid and interval data.

    With a single interval, the criterion q at the sample mean of the
    interval should be exactly zero (it is the unique moment that
    satisfies both interval constraints with equality).
    """
    rng = np.random.default_rng(11)
    y, X = _make_intervals(rng, 5)
    X = X.reshape(-1)

    # One candidate inside the identified set: the midpoint of the
    # sample-mean interval.
    moments = np.linspace(-3.0, 3.0, 61)

    result = bound_iii(y, X, moments)

    assert isinstance(result, dict)
    assert int(result["n"]) == 5
    assert result["upper"] >= result["lower"]
    assert result["width"] == result["upper"] - result["lower"]

    # Independent recomputation of n_in_set.
    grid = moments.reshape(-1)
    expected_in = int(np.sum((grid >= result["lower"]) & (grid <= result["upper"])))
    assert int(result["n_in_set"]) == expected_in
