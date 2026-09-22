"""Tests for bicsm.bic_score_dag."""

from morie.fn import _array_core as np

from morie.fn.bicsm import bic_score_dag


def test_bicsm_basic():
    """Test basic functionality."""
    dag = {"A": [], "B": ["A"], "C": ["B"]}
    rng = np.random.default_rng(42)
    n = 100
    a = rng.normal(0, 1, n)
    b = 0.5 * a + rng.normal(0, 1, n)
    c = -0.3 * b + rng.normal(0, 1, n)
    data = np.column_stack([a, b, c])
    names = ["A", "B", "C"]
    result = bic_score_dag(data, dag, names)
    assert hasattr(result, "payload") or isinstance(result, dict)
    payload = result.payload if hasattr(result, "payload") else result
    assert "score" in payload
    assert "loglik" in payload
    assert "k" in payload
    assert "penalty" in payload

    # Independent recomputation of the documented BIC formula.
    # k = 1 intercept + 1 slope per parent + 1 variance per node.
    k = 0
    for node, parents in dag.items():
        k += 1  # intercept
        k += len(parents)  # slopes
    k += len(dag)  # one variance per node

    # Residual variance per node from OLS of node on its parents.
    def _ols_resid_var(target, predictors):
        # predictors is an (n, p) array (empty column if no parents).
        n_obs = target.shape[0]
        if predictors.shape[1] == 0:
            resid = target - target.mean()
        else:
            X = np.column_stack([np.ones(n_obs), predictors])
            beta, *_ = np.linalg.lstsq(X, target, rcond=None)
            resid = target - X @ beta
        return float((resid @ resid) / n_obs)

    columns = {"A": 0, "B": 1, "C": 2}
    loglik = 0.0
    import math
    for node, parents in dag.items():
        y = data[:, columns[node]]
        X = data[:, [columns[p] for p in parents]] if parents else np.zeros((n, 0))
        s2 = _ols_resid_var(y, X)
        loglik += -n / 2.0 * (math.log(2.0 * math.pi * s2) + 1.0)
    penalty = 0.5 * math.log(n) * k
    expected_score = loglik - penalty

    assert payload["k"] == k
    assert math.isclose(payload["loglik"], loglik, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(payload["penalty"], penalty, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(payload["score"], expected_score, rel_tol=1e-9, abs_tol=1e-9)


def test_bicsm_edge():
    """Test edge cases: no edges (each node independent)."""
    dag = {"A": [], "B": [], "C": []}
    rng = np.random.default_rng(7)
    n = 50
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c = rng.normal(0, 1, n)
    data = np.column_stack([a, b, c])
    names = ["A", "B", "C"]
    result = bic_score_dag(data, dag, names)
    payload = result.payload if hasattr(result, "payload") else result
    assert "score" in payload
    assert "loglik" in payload
    assert "k" in payload
    assert "penalty" in payload
