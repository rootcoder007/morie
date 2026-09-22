"""Tests for dnnmt.dnn_multitrait."""

from morie.fn import _array_core as np

from morie.fn.dnnmt import dnn_multitrait


def _row(v):
    """Convert a 1D marr to a Python list of floats."""
    return [float(v[i]) for i in range(len(v))]


def _mat(arr):
    """Convert a 2D marr to a list-of-lists of floats."""
    return [_row(arr[i]) for i in range(len(arr))]


def test_dnnmt_basic():
    """Test basic functionality with deterministic random data."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    n, p, T = 100, 5, 3

    X_2d = rng_x.normal(0.0, 1.0, (n, p))
    Y_2d = rng_y.normal(0.0, 1.0, (n, T))

    X = _mat(X_2d)
    Y = _mat(Y_2d)

    layers = [8, 8]
    heads = [1.0, 1.0, 1.0]

    result = dnn_multitrait(
        X, Y, layers,
        heads=heads,
        activation="relu",
        out_activation="linear",
        eta=0.05,
        epochs=20,
        tol=0.0,
        seed=1,
    )

    # Documented return: n-by-T matrix of predictions after training.
    assert hasattr(result, "payload"), "result must expose a .payload"
    Y_hat = result.payload["Y_hat"]
    assert len(Y_hat) == n
    assert len(Y_hat[0]) == T
    for i in range(n):
        for t in range(T):
            assert isinstance(Y_hat[i][t], float)

    # Loss and epochs_run are documented payload keys.
    loss = result.payload["loss"]
    assert isinstance(loss, float)
    assert loss >= 0.0
    assert result.payload["epochs_run"] == 20

    # Head weights stored as the per-trait loss weights (one per output).
    head_weights = result.payload["head_weights"]
    assert len(head_weights) == T
    for w in head_weights:
        assert isinstance(w, float)


def test_dnnmt_edge():
    """Edge case: two observations, two traits, single hidden layer."""
    X_2d = np.random.default_rng(7).normal(0.0, 1.0, (2, 4))
    Y_2d = np.random.default_rng(8).normal(0.0, 1.0, (2, 2))

    X = _mat(X_2d)
    Y = _mat(Y_2d)

    layers = [3]
    result = dnn_multitrait(
        X, Y, layers,
        heads=None,
        activation="tanh",
        out_activation="linear",
        eta=0.1,
        epochs=5,
        tol=0.0,
        seed=2,
    )
    assert hasattr(result, "payload")
    Y_hat = result.payload["Y_hat"]
    assert len(Y_hat) == 2
    assert len(Y_hat[0]) == 2
    assert result.payload["epochs_run"] == 5
