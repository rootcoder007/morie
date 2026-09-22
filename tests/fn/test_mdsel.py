"""Tests for morie.fn.mdsel — model selection via CV loss."""

from morie.fn import _array_core as np
import pytest

from morie.fn.mdsel import mdsel


def test_basic_output():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 3))
    y = X[:, 0] + rng.standard_normal(n) * 0.5
    result = mdsel(X, y, n_folds=3, seed=7)
    assert "selected" in result
    assert "selected_1se" in result
    assert "cv_risks" in result
    assert "cv_ses" in result
    assert "model_names" in result
    assert "n" in result
    assert "n_folds" in result
    assert result["n"] == n
    assert result["n_folds"] == 3
    assert len(result["cv_risks"]) == 4  # default: Intercept, OLS, Ridge(1), Ridge(10)
    assert len(result["cv_ses"]) == 4
    assert len(result["model_names"]) == 4
    assert set(result["model_names"]) == {"Intercept", "OLS", "Ridge(1)", "Ridge(10)"}
    assert 0 <= result["selected"] < len(result["model_names"])
    assert 0 <= result["selected_1se"] < len(result["model_names"])


def test_ols_beats_intercept():
    rng = np.random.default_rng(42)
    n = 300
    X = rng.standard_normal((n, 2))
    y = 2.0 * X[:, 0] + rng.standard_normal(n) * 0.3
    result = mdsel(X, y, n_folds=5, seed=1)
    ols_idx = result["model_names"].index("OLS")
    intercept_idx = result["model_names"].index("Intercept")
    assert result["cv_risks"][ols_idx] < result["cv_risks"][intercept_idx]


def test_1se_rule():
    rng = np.random.default_rng(7)
    n = 200
    X = rng.standard_normal((n, 2))
    y = X[:, 0] + rng.standard_normal(n)
    result = mdsel(X, y, n_folds=3, seed=42)
    assert 0 <= result["selected_1se"] < len(result["model_names"])

    # The 1SE rule: selected_1se must satisfy cv_risks[selected_1se] <= cv_risks[selected] + cv_ses[selected]
    selected = result["selected"]
    threshold = float(result["cv_risks"][selected] + result["cv_ses"][selected])
    assert float(result["cv_risks"][result["selected_1se"]]) <= threshold


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        mdsel(np.zeros((0, 2)), np.zeros(0))
