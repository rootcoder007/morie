"""Tests for morie.fn.admod — Additive model via marginal integration."""

import pytest

from morie.fn import _array_core as np
from morie.fn.admod import admod


@pytest.fixture()
def synth():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 2))
    Y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.2 * rng.standard_normal(n)
    return Y, X


def test_returns_dict(synth):
    Y, X = synth
    result = admod(Y, X)
    assert isinstance(result, dict)
    for key in ("intercept", "components", "residuals", "n", "p", "method"):
        assert key in result


def test_components_count(synth):
    Y, X = synth
    result = admod(Y, X)
    assert len(result["components"]) == 2


def test_component_grid_size(synth):
    Y, X = synth
    result = admod(Y, X, grid_size=30)
    for comp in result["components"]:
        assert len(comp["x_grid"]) == 30
        assert len(comp["m_hat"]) == 30


def test_residuals_length(synth):
    Y, X = synth
    result = admod(Y, X)
    assert len(result["residuals"]) == 200


def test_residuals_small(synth):
    Y, X = synth
    result = admod(Y, X)
    assert np.std(result["residuals"]) < np.std(Y)


def test_method_label(synth):
    Y, X = synth
    result = admod(Y, X)
    assert result["method"] == "AdditiveModel_MarginalIntegration"


def test_component_is_the_pilot_averaged_over_the_other_covariate():
    """m_1(t) = mean_k g(t, X_k2) - ybar with the 2-D Nadaraya-Watson pilot g."""
    import math

    import pytest

    X = [[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0], [4.0, 3.0], [5.0, 2.0], [1.5, 2.5]]
    Y = [1.0, 1.5, 3.2, 3.9, 6.1, 6.0, 2.2]
    h = 1.3
    n = len(Y)
    ybar = sum(Y) / n

    def g(t1, t2):
        w = [math.exp(-0.5 * ((t1 - x[0]) / h) ** 2 - 0.5 * ((t2 - x[1]) / h) ** 2) for x in X]
        return sum(wi * yi for wi, yi in zip(w, Y)) / sum(w)

    r = admod(Y, X, bandwidth=h, grid_size=4)
    grid = r["components"][0]["x_grid"]
    assert grid == pytest.approx([0.0, 5 / 3, 10 / 3, 5.0], rel=1e-15)
    want = [sum(g(t, x[1]) for x in X) / n - ybar for t in grid]
    assert r["components"][0]["m_hat"] == pytest.approx(want, rel=1e-12, abs=1e-13)
    want2 = [sum(g(x[0], t) for x in X) / n - ybar for t in r["components"][1]["x_grid"]]
    assert r["components"][1]["m_hat"] == pytest.approx(want2, rel=1e-12, abs=1e-13)
    assert r["intercept"] == pytest.approx(ybar, rel=1e-15)
