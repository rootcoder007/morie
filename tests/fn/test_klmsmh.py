"""Tests for klmsmh.kalman_smoother."""

import math

from morie.fn import _array_core as np
from morie.fn.klmsmh import kalman_smoother


def _make_inputs(n=10, d=2, seed=42):
    """Build valid (y, model, filtered) inputs for kalman_smoother."""
    rng_y = np.random.default_rng(seed + 1)
    y = list(rng_y.normal(0, 1, n))

    rng_m = np.random.default_rng(seed)
    F = [[rng_m.normal(0, 1) for _ in range(d)] for _ in range(d)]
    model = {"F": F}

    rng_f = np.random.default_rng(seed + 2)
    eye = [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]
    cov_half = [[0.5 if i == j else 0.0 for j in range(d)] for i in range(d)]

    states = []
    covs = []
    predicted = []
    predicted_cov = []

    for t in range(n):
        if t == 0:
            x_pred = [0.0] * d
            P_pred = [r[:] for r in eye]
        else:
            x_prev = states[t - 1]
            P_prev = covs[t - 1]
            x_pred = [sum(F[i][k] * x_prev[k] for k in range(d)) for i in range(d)]
            P_pred = [
                [
                    sum(F[i][k] * sum(P_prev[k][l] * F[j][l] for l in range(d)) for k in range(d))
                    for j in range(d)
                ]
                for i in range(d)
            ]
        predicted.append(x_pred)
        predicted_cov.append(P_pred)
        x_filt = [x_pred[i] + 0.05 * rng_f.normal(0, 1) for i in range(d)]
        states.append(x_filt)
        covs.append([r[:] for r in cov_half])

    filtered = {
        "state": states,
        "cov": covs,
        "predicted": predicted,
        "predicted_cov": predicted_cov,
    }

    return y, model, filtered


def test_klmsmh_basic():
    """Test basic functionality."""
    y, model, filtered = _make_inputs(n=10, d=2, seed=42)
    result = kalman_smoother(y, model, filtered)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "smoothed" in result
    assert "smoothed_cov" in result
    assert "n" in result
    assert "method" in result

    assert result["n"] == 10
    assert math.isfinite(float(result["estimate"]))
    assert len(result["smoothed"]) == 10
    assert len(result["smoothed_cov"]) == 10


def test_klmsmh_edge():
    """Test edge cases."""
    y, model, filtered = _make_inputs(n=5, d=2, seed=7)
    result = kalman_smoother(y, model, filtered, ridge=1e-8)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "smoothed" in result
    assert result["n"] == 5
    assert math.isfinite(float(result["estimate"]))
    assert len(result["smoothed"]) == 5
