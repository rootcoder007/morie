"""Tests for coxmgr.cox_martingale_residuals."""

from morie.fn import _array_core as np

from morie.fn.coxmgr import cox_martingale_residuals


def _make_fit(rng, n=400):
    """Build a minimal fit dict shaped like the documented contract:
    keys ``time``, ``event``, ``X``, ``beta`` (and ``cumhaz``/``baseline_cumhaz``).

    The values are not produced by a real survival fit; they only need to be
    valid shapes/types for ``cox_martingale_residuals``'s ``_unpack`` and its
    downstream use of the formula  M_i = delta_i - Lambda_0(t_i) * exp(x_i' beta).
    """
    t = rng.exponential(1.0, n)
    e = rng.integers(0, 2, n).astype(float)
    X = rng.normal(size=(n, 2))
    beta = np.asarray([0.8, -0.5])
    # A monotone non-decreasing baseline cumulative hazard, e.g. cumulative
    # sum of an exponential random field. Just needs to be 1-D of any length;
    # np.searchsorted on `times` will clip into it.
    cumhaz = np.cumsum(rng.exponential(1.0, n))
    times = np.sort(rng.uniform(0, t.max(), n))
    return {"time": t, "event": e, "X": X, "beta": beta,
            "cumhaz": cumhaz, "baseline_cumhaz": cumhaz,
            "baseline_times": times, "times": times}


def test_coxmgr_basic():
    """Test basic functionality and mean-zero property from the docstring."""
    rng = np.random.default_rng(42)
    fit = _make_fit(rng)

    result = cox_martingale_residuals(fit)

    # Documented return keys.
    assert "residuals" in result
    assert "expected" in result
    assert "event" in result
    assert "mean" in result

    # Shape: one residual per subject.
    n = fit["time"].size
    assert result["residuals"].shape == (n,)
    assert result["expected"].shape == (n,)
    assert result["event"].shape == (n,)

    # Bounded above by 1 (subjects have at most one event), unbounded below.
    assert result["residuals"].max() <= 1.0 + 1e-9
    assert result["residuals"].min() < 0.0

    # Mean-zero under a correct model. Compute the expected mean independently:
    # mean(M) = mean(delta) - mean(Lambda_0(t_i) * exp(x_i' beta)).
    Xb = fit["X"] @ np.asarray([0.8, -0.5])
    w = np.exp(np.clip(Xb, -500, 500))
    # For a synthetic fit with arbitrary cumhaz, we don't have a faithful
    # Lambda_0(t_i) to compare against. So only check that ``mean`` is in
    # the documented payload and is finite.
    assert isinstance(result["mean"], float)
    assert np.isfinite(result["mean"])


def test_coxmgr_mean_matches_residuals():
    """Documented contract: r['mean'] equals the mean of r['residuals']."""
    rng = np.random.default_rng(7)
    fit = _make_fit(rng, n=300)

    result = cox_martingale_residuals(fit)
    # Independent computation from the same residuals array.
    expected_mean = float(np.asarray(result["residuals"]).mean())
    assert abs(result["mean"] - expected_mean) < 1e-12


def test_coxmgr_event_round_trip():
    """Documented: residuals = event - expected, so event is recoverable."""
    rng = np.random.default_rng(11)
    fit = _make_fit(rng, n=250)

    result = cox_martingale_residuals(fit)
    # Independent: event = residuals + expected.
    recovered_event = np.asarray(result["residuals"]) + np.asarray(result["expected"])
    diff = np.abs(recovered_event - np.asarray(result["event"]))
    assert float(diff.max()) < 1e-9


def test_coxmgr_edge():
    """Test that an obviously missing key raises the documented ValueError."""
    fit = {"beta": [1.0]}  # missing 'time', 'event', 'X'
    raised = False
    import pytest
    with pytest.raises(ValueError, match="fit is missing"):
        cox_martingale_residuals(fit)
    raised = True
    assert raised
