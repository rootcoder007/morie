"""Tests for ghs001.ghosal_ch1_bayes_formula."""

from morie.fn import _array_core as np

from morie.fn.ghs001 import ghosal_ch1_bayes_formula


def _make_inputs(seed_supp=0, seed_wts=1, seed_X=2):
    supp = np.random.default_rng(seed_supp).normal(0, 1, 10)
    wts = np.abs(np.random.default_rng(seed_wts).normal(0, 1, 10))
    wts = wts / wts.sum()
    X = np.random.default_rng(seed_X).normal(0, 1, 5)
    return supp, wts, X


def test_ghosal_ch1_bayes_formula_basic():
    """Test basic functionality with the correct arity and argument shapes."""
    supp, wts, X = _make_inputs()

    def p_theta(t, x):
        return float(np.exp(-0.5 * float(np.sum((x - t) ** 2))))

    def B(t):
        return float(t) < 0.0

    Pi = (supp, wts)
    result = ghosal_ch1_bayes_formula(B, X, p_theta, Pi)

    assert hasattr(result, "payload")
    assert "posterior" in result.payload
    assert "marginal" in result.payload
    assert result.payload["method"].startswith("Bayes formula")

    # Independent re-derivation from the documented formula.
    liks = [p_theta(t, X) for t in supp]
    expected_num = sum(l * w for l, w, t in zip(liks, wts, supp) if B(t))
    expected_den = sum(l * w for l, w in zip(liks, wts))
    expected_posterior = expected_num / expected_den
    expected_marginal = expected_den

    assert result.payload["posterior"] == expected_posterior
    assert result.payload["marginal"] == expected_marginal


def test_ghosal_ch1_bayes_formula_edge_full_support():
    """When B is the full support indicator, posterior equals 1 by construction."""
    supp, wts, X = _make_inputs(seed_supp=3, seed_wts=4, seed_X=5)

    def p_theta(t, x):
        return float(np.exp(-0.5 * float(np.sum((x - t) ** 2))))

    def B(t):
        return True

    Pi = (supp, wts)
    result = ghosal_ch1_bayes_formula(B, X, p_theta, Pi)

    liks = [p_theta(t, X) for t in supp]
    expected_marginal = sum(l * w for l, w in zip(liks, wts))
    assert result.payload["posterior"] == 1.0
    assert result.payload["marginal"] == expected_marginal
