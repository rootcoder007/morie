"""Tests for coxres.cox_schoenfeld_residuals."""

from morie.fn import _array_core as np

from morie.fn.coxres import cox_schoenfeld_residuals


def _make_fit(n=120, p=1, seed=42):
    """Build a minimal Cox fit mapping matching the documented interface."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, p))
    beta = rng.normal(0, 0.3, p)
    T = rng.exponential(1.0 / np.exp(X @ beta), n)
    C = rng.exponential(3.0, n)
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    return {"time": time, "event": event, "X": X, "beta": beta}


def test_coxres_basic():
    """Basic test: returns a mapping with the documented keys."""
    fit = _make_fit(seed=42)
    result = cox_schoenfeld_residuals(fit)
    assert isinstance(result, dict)
    assert "residuals" in result
    assert "times" in result
    assert "correlation" in result
    assert "p_value" in result
    assert "transform" in result
    assert int(result["residuals"].shape[0]) == int(np.asarray(fit["event"]).sum())
    assert result["transform"] == "km"


def test_coxres_mean_zero():
    """Under a roughly correct fit, Schoenfeld residuals have ~zero mean."""
    fit = _make_fit(seed=123)
    result = cox_schoenfeld_residuals(fit)
    assert abs(float(np.asarray(result["residuals"]).mean())) < 0.5


def test_coxres_ph_holds():
    """Generated data respects PH, so the trend test should not fire (p > 0.05)."""
    fit = _make_fit(seed=7)
    result = cox_schoenfeld_residuals(fit)
    assert float(np.asarray(result["p_value"])[0]) > 0.05


def test_coxres_transforms_run():
    """All three documented transforms produce the same residual set."""
    fit = _make_fit(seed=0)
    for tr in ("km", "rank", "identity"):
        result = cox_schoenfeld_residuals(fit, transform=tr)
        assert result["transform"] == tr
        assert int(result["residuals"].shape[0]) == int(np.asarray(fit["event"]).sum())


def test_coxres_formula_manual():
    """Verify the residual formula against an independent computation."""
    fit = _make_fit(seed=4)
    t = np.asarray(fit["time"], dtype=float)
    e = np.asarray(fit["event"], dtype=float)
    X = np.asarray(fit["X"], dtype=float)
    beta = np.asarray(fit["beta"], dtype=float).ravel()
    w = np.exp(X @ beta)

    ev_idx = [i for i in range(t.size) if e[i] == 1]
    # stable order by time
    ev_idx = sorted(ev_idx, key=lambda i: t[i])

    expected = []
    for i in ev_idx:
        at_risk = [k for k in range(t.size) if t[k] >= t[i]]
        wr_sum = sum(w[k] for k in at_risk)
        num = sum(w[k] * X[k] for k in at_risk)
        expected.append(X[i] - num / wr_sum)

    result = cox_schoenfeld_residuals(fit)
    got = np.asarray(result["residuals"])
    for r_i, i in enumerate(ev_idx):
        assert abs(float(got[r_i, 0]) - float(expected[r_i][0])) < 1e-10


def test_coxres_edge():
    """Test edge cases: small fit still returns the documented keys."""
    fit = _make_fit(n=40, seed=1)
    result = cox_schoenfeld_residuals(fit)
    assert isinstance(result, dict)
    assert "residuals" in result
    assert "p_value" in result
