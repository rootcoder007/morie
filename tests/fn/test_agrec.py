"""Tests for agrec.andersen_gill_recurrent."""

from morie.fn import _array_core as np

from morie.fn.agrec import andersen_gill_recurrent


def _build_interval_data(n_subjects=50, n_int_per=3, seed=0):
    """Build valid counting-process data: intervals with stop > start and binary events."""
    rng = np.random.default_rng(seed)
    n_total = n_subjects * n_int_per
    # positive interval widths drawn from exponential
    widths = rng.exponential(scale=2.0, size=n_total)
    starts = np.zeros(n_total)
    stops = np.zeros(n_total)
    events = np.zeros(n_total, dtype=np.int64)
    for i in range(n_subjects):
        t = 0.0
        for j in range(n_int_per):
            idx = i * n_int_per + j
            starts[idx] = t
            t = t + widths[idx]
            stops[idx] = t
            # 70% of intervals end in an event
            events[idx] = 1 if rng.random() < 0.7 else 0
    X = rng.normal(0.0, 1.0, size=(n_total, 5))
    return starts, stops, events, X


def test_agrec_basic():
    """Test basic functionality."""
    start, stop, event, X = _build_interval_data()
    result = andersen_gill_recurrent(start, stop, event, X)

    # The function returns a RichResult; check the documented keys.
    assert "estimate" in result
    assert "se" in result
    assert "cov" in result
    assert "loglik" in result
    assert "n_iter" in result
    assert "n_events" in result
    assert result["method"].startswith("Andersen-Gill")

    # Shapes: estimate and se have one entry per covariate; cov is (p, p).
    estimate = np.asarray(result["estimate"])
    se = np.asarray(result["se"])
    cov = np.asarray(result["cov"])
    n_events = int(result["n_events"])

    assert estimate.shape == (X.shape[1],)
    assert se.shape == (X.shape[1],)
    assert cov.shape == (X.shape[1], X.shape[1])

    # n_events matches the number of rows whose event indicator is 1.
    assert n_events == int(np.sum(event))

    # Sample size used by the estimator matches the row count of X.
    n_total = X.shape[0]
    # Standard errors must be finite and non-negative.
    assert np.all(np.isfinite(se))
    assert np.all(se >= 0.0)

    # Cross-check: se^2 should agree (up to tolerance) with the diagonal
    # of the covariance matrix.
    diff = np.max(np.abs(np.asarray(se) ** 2 - np.diag(cov)))
    assert float(diff) < 1e-8

    # Independent recomputation of the partial log-likelihood at the
    # returned estimate, using the Breslow formula documented in the
    # function's docstring:
    #   L(beta) = sum_{intervals k with event_k=1}
    #             [ beta' x_k - log( sum_j Y_j(t_k) exp(beta' x_j) ) ]
    b = np.asarray(estimate)
    # Per-interval linear predictor.
    lp_all = X @ b
    # Event times in insertion order (assumes docstring's counting-process
    # ordering, one row per interval).
    loglik_terms = []
    for k in range(n_total):
        if int(event[k]) != 1:
            continue
        t_k = float(stop[k])
        # At-risk set: all intervals with start < t_k <= stop, i.e. the
        # subject is under observation at t_k.
        at_risk = (np.asarray(stop) >= t_k) & (np.asarray(start) < t_k)
        if int(np.sum(at_risk)) == 0:
            continue
        contrib = float(lp_all[k]) - float(np.log(np.sum(np.exp(lp_all[at_risk]))))
        loglik_terms.append(contrib)
    expected_loglik = float(np.sum(loglik_terms))
    assert abs(float(result["loglik"]) - expected_loglik) < 1e-6


def test_agrec_edge():
    """Test edge cases."""
    start, stop, event, X = _build_interval_data(n_subjects=20, n_int_per=2, seed=1)
    result = andersen_gill_recurrent(start, stop, event, X)

    assert "estimate" in result
    assert "se" in result
    assert "cov" in result
    assert "loglik" in result
    assert "n_iter" in result
    assert "n_events" in result

    estimate = np.asarray(result["estimate"])
    se = np.asarray(result["se"])
    assert estimate.shape == (X.shape[1],)
    assert se.shape == (X.shape[1],)

    # The fit converged within max_iter and event count matches the data.
    assert int(result["n_iter"]) <= 50
    assert int(result["n_events"]) == int(np.sum(event))
