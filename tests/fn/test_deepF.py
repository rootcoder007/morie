"""Tests for deepF.deepfm."""

import math

from morie.fn import _array_core as np

from morie.fn.deepF import deepfm


def _to_rows(X, n, p):
    """Convert X marr/array-like to a plain list of n lists of p floats."""
    rows = []
    for i in range(n):
        row = [float(X[i][j]) for j in range(p)]
        rows.append(row)
    return rows


def _independent_fm(X_rows, w, V, w0, p, K):
    """Independent FM computation matching the documented formula."""
    out = []
    for x in X_rows:
        lin = w0 + sum(w[j] * x[j] for j in range(p))
        # FM second-order term: 0.5 * sum_f ((sum_j v_jf x_j)^2 - sum_j v_jf^2 x_j^2)
        s = 0.0
        s2 = 0.0
        for f in range(K):
            acc = 0.0
            acc2 = 0.0
            for j in range(p):
                acc += V[j][f] * x[j]
                acc2 += (V[j][f] ** 2) * (x[j] ** 2)
            s += acc * acc - acc2
        wide = lin + 0.5 * s
        out.append(wide)
    return out


def test_deepF_basic():
    """Test basic functionality with documented argument shapes/types."""
    rng_X = np.random.default_rng(42)
    X = rng_X.normal(0, 1, (20, 5))
    # y must be binary 0/1 per docstring
    y = [1.0 if i % 2 == 0 else 0.0 for i in range(20)]

    # K and mlp_h are ints (embedding dim and hidden width)
    K = 4
    mlp_h = 4
    seed = 42
    w0 = 0.0
    deep_scale = 0.0  # reduces prediction to pure FM (documented behaviour)

    # Build X as a plain list of lists to feed core.mat cleanly
    n, p = 20, 5
    X_rows = _to_rows(X, n, p)

    result = deepfm(X_rows, y, K, mlp_h, w0, seed, deep_scale)

    assert isinstance(result, dict)
    # Documented keys (per docstring)
    for key in ("estimate", "p_hat", "fm_part", "deep_part",
                "logloss", "n", "p", "K"):
        assert key in result, f"missing documented key {key!r}"
    assert result["method"] == "DeepFM: factorization machine plus deep network"
    assert result["n"] == n
    assert result["p"] == p
    assert result["K"] == K
    assert len(result["p_hat"]) == n
    assert len(result["fm_part"]) == n
    assert len(result["deep_part"]) == n
    # deep_scale=0 -> deep_part is identically zero
    assert all(abs(v) < 1e-12 for v in result["deep_part"])

    # Independent expected computation (FM only, since deep_scale=0).
    # Replay the same deterministic stream the function uses to obtain w, V.
    rng = np.random.default_rng(seed)
    w = [float(rng.normal(0.0, 0.1)) for _ in range(p)]
    V = [[float(rng.normal(0.0, 0.1)) for _ in range(K)] for _ in range(p)]

    expected_fm = _independent_fm(X_rows, w, V, w0, p, K)
    expected_ph = [1.0 / (1.0 + math.exp(-z)) for z in expected_fm]
    expected_estimate = sum(expected_ph) / n
    expected_ll = -sum(
        y[i] * math.log(expected_ph[i] + 1e-300)
        + (1 - y[i]) * math.log(1 - expected_ph[i] + 1e-300)
        for i in range(n)
    ) / n

    for i in range(n):
        assert abs(result["fm_part"][i] - expected_fm[i]) < 1e-9
        assert abs(result["p_hat"][i] - expected_ph[i]) < 1e-9

    assert abs(result["estimate"] - expected_estimate) < 1e-9
    assert abs(result["logloss"] - expected_ll) < 1e-9
    # Estimates lie in (0, 1)
    assert 0.0 < result["estimate"] < 1.0


def test_deepF_edge():
    """Test edge cases with documented argument shapes/types."""
    rng_X = np.random.default_rng(42)
    X = rng_X.normal(0, 1, (10, 3))
    y = [0.0, 1.0] * 5  # valid binary labels

    K = 2
    mlp_h = 3
    seed = 7
    w0 = 0.1
    deep_scale = 1.0

    n, p = 10, 3
    X_rows = _to_rows(X, n, p)

    result = deepfm(X_rows, y, K, mlp_h, w0, seed, deep_scale)

    assert isinstance(result, dict)
    for key in ("estimate", "p_hat", "fm_part", "deep_part",
                "logloss", "n", "p", "K", "method"):
        assert key in result
    assert result["n"] == n
    assert result["p"] == p
    assert result["K"] == K
    assert len(result["p_hat"]) == n
    assert len(result["fm_part"]) == n
    assert len(result["deep_part"]) == n
    assert 0.0 < result["estimate"] < 1.0
    # logloss is finite when y is provided and binary
    assert math.isfinite(result["logloss"])
