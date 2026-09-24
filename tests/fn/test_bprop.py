"""Tests for bprop.backpropagation_chain_rule."""

import math

from morie.fn import _array_core as np

from morie.fn.bprop import backpropagation_chain_rule


def test_bprop_basic():
    """Test basic functionality with a small 3-layer feedforward network."""
    rng = np.random.default_rng(42)
    n = 5        # number of patterns
    d_in = 3     # input dimension
    h1 = 4       # hidden layer 1 units
    h2 = 3       # hidden layer 2 units
    d_out = 2    # output dimension

    # W_l has one row per unit of layer l and one column per unit of
    # layer l-1 plus a leading bias column.
    W1 = rng.normal(0, 1, (h1, d_in + 1))
    W2 = rng.normal(0, 1, (h2, h1 + 1))
    W3 = rng.normal(0, 1, (d_out, h2 + 1))
    layers = [W1, W2, W3]

    # Forward activations: a_0 (input, no bias column) through a_L (output).
    a0 = rng.normal(0, 1, (n, d_in))
    a1 = rng.normal(0, 1, (n, h1))
    a2 = rng.normal(0, 1, (n, h2))
    a3 = rng.normal(0, 1, (n, d_out))
    activations = [a0, a1, a2, a3]

    # loss_grad: n-by-units_L matrix of dE/dyhat.
    loss_grad = rng.normal(0, 1, (n, d_out))

    result = backpropagation_chain_rule(layers, activations, loss_grad)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "gradients" in result
    assert "deltas" in result

    # estimate is gradients[0][0][0]: a single scalar.
    assert math.isfinite(result["estimate"])

    # gradients: one increment matrix per layer, shaped like layers.
    assert isinstance(result["gradients"], list)
    assert len(result["gradients"]) == len(layers)
    for g, W in zip(result["gradients"], layers):
        assert len(g) == len(W)
        assert len(g[0]) == len(W[0])

    # deltas: one per layer; deltas[-1] = delta, deltas[0] = psi.
    assert isinstance(result["deltas"], list)
    assert len(result["deltas"]) == len(layers)
    assert len(result["deltas"][-1]) == n
    assert len(result["deltas"][-1][0]) == d_out
    assert len(result["deltas"][0]) == n
    assert len(result["deltas"][0][0]) == h1


def test_bprop_edge():
    """Test edge case with a minimal 1-layer, 1-pattern network."""
    rng = np.random.default_rng(7)
    n = 1
    d_in = 2
    d_out = 1

    W1 = rng.normal(0, 1, (d_out, d_in + 1))
    layers = [W1]

    a0 = rng.normal(0, 1, (n, d_in))
    a1 = rng.normal(0, 1, (n, d_out))
    activations = [a0, a1]

    loss_grad = rng.normal(0, 1, (n, d_out))

    result = backpropagation_chain_rule(layers, activations, loss_grad)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "gradients" in result
    assert "deltas" in result
    assert len(result["gradients"]) == 1
    assert len(result["gradients"][0]) == d_out
    assert len(result["gradients"][0][0]) == d_in + 1
    assert math.isfinite(result["estimate"])
