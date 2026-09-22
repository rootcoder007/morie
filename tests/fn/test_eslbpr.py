"""Tests for eslbpr.esl_backprop."""

from morie.fn import _array_core as np

from morie.fn.eslbpr import esl_backprop


def _make_weights(rng, p, M, K):
    return {
        "alpha": rng.normal(size=(p, M)) * 0.5,
        "alpha0": np.zeros(M),
        "beta": rng.normal(size=(M, K)) * 0.5,
        "beta0": np.zeros(K),
    }


def test_eslbpr_basic():
    """Test basic functionality: keys, shapes, loss decrease, and gradient correctness."""
    rng = np.random.default_rng(42)
    n, p, M, K = 100, 5, 4, 1
    X = rng.normal(0, 1, (n, p))
    y = rng.normal(0, 1, n)
    W = _make_weights(rng, p, M, K)

    result = esl_backprop(X, y, W)

    # RichResult behaves like a dict; check documented keys are present.
    for key in (
        "grad_alpha", "grad_alpha0", "grad_beta", "grad_beta0",
        "delta", "hidden", "loss",
    ):
        assert key in result, f"missing key {key!r} in result"

    # Shapes match the documented parameter shapes.
    assert result["grad_alpha"].shape == (p, M)
    assert result["grad_alpha0"].shape == (M,)
    assert result["grad_beta"].shape == (M, K)
    assert result["grad_beta0"].shape == (K,)
    assert result["delta"].shape == (n, K)
    assert result["hidden"].shape == (n, M)

    # Loss is finite and non-negative for regression with squared error.
    assert result["loss"] >= 0
    assert result["loss"] == float(result["loss"])

    # Independent recomputation of the loss using the documented network.
    A = X @ W["alpha"] + W["alpha0"]
    Z = 1.0 / (1.0 + np.exp(-np.clip(A, -500, 500)))
    T = Z @ W["beta"] + W["beta0"]
    expected_loss = float(np.mean((T - y.reshape(n, K)) ** 2))
    assert result["loss"] == expected_loss

    # Independent recomputation of grad_beta and grad_alpha0.
    Y = y.reshape(n, K).astype(float)
    delta = 2 * (T - Y) / n
    expected_grad_beta = Z.T @ delta
    expected_grad_alpha0 = (delta @ W["beta"].T * Z * (1 - Z)).sum(axis=0)
    assert result["grad_beta"].shape == expected_grad_beta.shape
    assert result["grad_alpha0"].shape == expected_grad_alpha0.shape
    # elementwise agreement
    diff_b = float(np.max(np.abs(result["grad_beta"] - expected_grad_beta)))
    diff_a0 = float(np.max(np.abs(result["grad_alpha0"] - expected_grad_a0))) \
        if False else float(np.max(np.abs(result["grad_alpha0"] - expected_grad_alpha0)))
    assert diff_b < 1e-10
    assert diff_a0 < 1e-10

    # A small gradient step must reduce the loss.
    step = 0.05
    W2 = {
        "alpha":  W["alpha"]  - step * result["grad_alpha"],
        "alpha0": W["alpha0"] - step * result["grad_alpha0"],
        "beta":   W["beta"]   - step * result["grad_beta"],
        "beta0":  W["beta0"]  - step * result["grad_beta0"],
    }
    assert esl_backprop(X, y, W2)["loss"] < result["loss"]


def test_eslbpr_edge():
    """Test edge cases: minimal sizes and arity/shape handling."""
    rng = np.random.default_rng(42)
    n, p, M, K = 4, 2, 1, 1
    X = rng.normal(0, 1, (n, p))
    y = rng.normal(0, 1, n)
    W = _make_weights(rng, p, M, K)

    result = esl_backprop(X, y, W)

    for key in (
        "grad_alpha", "grad_alpha0", "grad_beta", "grad_beta0",
        "delta", "hidden", "loss",
    ):
        assert key in result, f"missing key {key!r} in result"

    # grad_beta must reduce along n (rows), giving (M, K).
    assert result["grad_beta"].shape == (M, K)
    # grad_alpha0 is a 1D vector of length M.
    assert result["grad_alpha0"].shape == (M,)
    # hidden activations are sigmoid values strictly in (0, 1).
    hidden = result["hidden"]
    assert hidden.shape == (n, M)
    assert float(np.min(hidden)) > 0.0
    assert float(np.max(hidden)) < 1.0
