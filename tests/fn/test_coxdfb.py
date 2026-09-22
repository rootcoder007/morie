"""Tests for coxdfb.cox_dfbeta_influence."""

from morie.fn import _array_core as np

from morie.fn.coxdfb import cox_dfbeta_influence


def _make_fit(n=60, p=2, seed=42):
    """Build a synthetic Cox fit mapping as expected by cox_dfbeta_influence."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.array([0.8, -0.5])
    T = rng.exponential(1.0 / np.exp(X @ beta))
    C = rng.exponential(2.0, n)
    e = (T <= C).astype(float)
    t = np.minimum(T, C)

    n_events = int(e.sum())
    I = rng.normal(size=(p, p))
    I = I + I.T + n * np.eye(p)
    se = rng.uniform(0.1, 1.0, size=p)

    return {
        "time": t,
        "event": e,
        "X": X,
        "beta": beta,
        "information": I,
        "se": se,
        "n_events": n_events,
    }


def test_coxdfb_basic():
    """Test basic functionality."""
    fit = _make_fit(n=60, p=2, seed=42)
    result = cox_dfbeta_influence(fit)

    # The function returns a RichResult that behaves like a dict.
    assert isinstance(result, dict)

    # Documented return keys.
    assert "dfbeta" in result
    assert "dfbetas" in result
    assert "max_influence" in result
    assert "most_influential" in result

    # Shape: (n, p) for dfbeta / dfbetas.
    n, p = 60, 2
    assert result["dfbeta"].shape == (n, p)
    assert result["dfbetas"].shape == (n, p)

    # SE-scaled dfbetas must equal dfbeta / se (elementwise over columns).
    se = np.asarray(fit["se"], dtype=float).ravel()
    expected_dfbetas = result["dfbeta"] / se
    assert np.allclose(result["dfbetas"], expected_dfbetas)

    # max_influence must equal max |dfbetas| (max over both axes).
    expected_max = float(np.max(np.abs(result["dfbetas"])))
    assert result["max_influence"] == expected_max

    # most_influential must be the subject index achieving max |dfbetas|.
    row_max = np.max(np.abs(result["dfbetas"]), axis=1)
    expected_worst = int(np.argmax(row_max))
    assert result["most_influential"] == expected_worst


def test_coxdfb_edge():
    """Test edge cases: well-behaved data, no single subject dominates."""
    fit = _make_fit(n=80, p=2, seed=7)
    result = cox_dfbeta_influence(fit)

    assert isinstance(result, dict)
    assert result["dfbeta"].shape == (80, 2)
    assert result["dfbetas"].shape == (80, 2)
    # In well-behaved simulated data no dfbetas value should reach 1.0.
    assert float(np.max(np.abs(result["dfbetas"]))) < 1.0
