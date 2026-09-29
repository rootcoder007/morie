"""Tests for morie.fn.bkfit — Backfitting algorithm for additive models."""

import pytest

from morie.fn import _array_core as np
from morie.fn.bkfit import bkfit


@pytest.fixture()
def synth():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 2))
    Y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.2 * rng.standard_normal(n)
    return Y, X


def test_returns_dict(synth):
    Y, X = synth
    result = bkfit(Y, X)
    assert isinstance(result, dict)
    for key in ("intercept", "components", "fitted", "residuals", "iterations", "converged", "n", "p", "method"):
        assert key in result


def test_components_count(synth):
    Y, X = synth
    result = bkfit(Y, X)
    assert len(result["components"]) == 2


def test_converged(synth):
    Y, X = synth
    result = bkfit(Y, X, max_iter=200)
    assert result["converged"]


def test_fitted_plus_resid(synth):
    Y, X = synth
    result = bkfit(Y, X)
    recon = result["fitted"] + result["residuals"]
    np.testing.assert_allclose(recon, Y, atol=1e-10)


def test_residuals_smaller_than_y(synth):
    Y, X = synth
    result = bkfit(Y, X)
    assert np.std(result["residuals"]) < np.std(Y)


def test_method_label(synth):
    Y, X = synth
    result = bkfit(Y, X)
    assert result["method"] == "Backfitting"


def test_one_backfitting_sweep_recomputed():
    """Gauss-Seidel sweep: smooth the partial residual on X_j, centre, update."""
    import math

    X = [[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0], [4.0, 3.0], [5.0, 2.0], [1.5, 2.5], [2.5, 0.5]]
    Y = [1.0, 1.5, 3.2, 3.9, 6.1, 6.0, 2.2, 2.8]
    n, h = 8, 1.1

    def smooth(col, v):
        out = []
        for i in range(n):
            w = [math.exp(-0.5 * ((col[k] - col[i]) / h) ** 2) for k in range(n)]
            out.append(sum(a * b for a, b in zip(w, v)) / sum(w))
        m = sum(out) / n
        return [o - m for o in out]

    a = sum(Y) / n
    c0 = smooth([r[0] for r in X], [Y[i] - a for i in range(n)])
    c1 = smooth([r[1] for r in X], [Y[i] - a - c0[i] for i in range(n)])
    a1 = sum(Y[i] - c0[i] - c1[i] for i in range(n)) / n
    r = bkfit(Y, X, bandwidth=h, max_iter=1)
    assert [float(v) for v in r["components"][0]] == pytest.approx(c0, rel=1e-12, abs=1e-14)
    assert [float(v) for v in r["components"][1]] == pytest.approx(c1, rel=1e-12, abs=1e-14)
    assert r["intercept"] == pytest.approx(a1, rel=1e-13)
