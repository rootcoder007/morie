# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deep (stacked) RNN: multiple recurrent layers."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_deep_rnn"]


def geron_deep_rnn(X, hidden_sizes=(4,), n_layers=None, weights=None, seed=0, activation="tanh"):
    """
    Deep (stacked) RNN: multiple recurrent layers.

    Formula: h_t^(l) = phi(W_x^(l) h_t^(l-1) + W_h^(l) h_{t-1}^(l))

    A real forward pass over the stack. Layer ``l`` consumes the sequence
    of hidden states produced by layer ``l-1`` -- so depth is over
    features, not over time -- and each layer carries its own recurrent
    state across time. Both loops are explicit here because the order
    matters: for a given time step every layer is updated bottom-up before
    time advances.

    Weights may be supplied; otherwise they are drawn from a deterministic
    LCG with standard deviation ``1/sqrt(fan_in)``. ``state_norms`` reports
    ``||h_t^(l)||`` per layer, which is where a stacked RNN's problems
    become visible: with ``tanh`` the norms saturate rather than explode,
    the trade-off that motivated LSTMs.

    Parameters
    ----------
    X : array-like, shape (T, d)
        Input sequence, one row per time step.
    hidden_sizes : int or sequence of int, default (4,)
        Width of each layer.
    n_layers : int, optional
        If given with a scalar ``hidden_sizes``, repeat that width.
    weights : sequence of (Wx, Wh, b), optional
        Explicit parameters per layer.
    seed : int, default 0
    activation : {"tanh", "relu"}, default "tanh"

    Returns
    -------
    result : RichResult
        Keys: outputs, states, final_states, layer_sizes, n_params,
        state_norms, n_layers, estimate, n, method.

    Examples
    --------
    A one-layer identity RNN accumulates its input over time exactly, with
    ReLU keeping the sum linear on non-negative data:

    >>> W = [([[1.0]], [[1.0]], [0.0])]
    >>> r = geron_deep_rnn([[1.0], [1.0], [1.0]], weights=W, activation="relu")
    >>> [row[0] for row in r["outputs"]]
    [1.0, 2.0, 3.0]
    >>> r["final_states"]
    [[3.0]]

    Two stacked identity layers integrate twice -- 1, 3, 6 are the
    triangular numbers:

    >>> W2 = [([[1.0]], [[1.0]], [0.0]), ([[1.0]], [[1.0]], [0.0])]
    >>> r2 = geron_deep_rnn([[1.0], [1.0], [1.0]], weights=W2, activation="relu")
    >>> [row[0] for row in r2["outputs"]]
    [1.0, 3.0, 6.0]
    >>> r2["n_layers"], r2["layer_sizes"]
    (2, [1, 1])

    tanh saturates instead of exploding, so the state stays bounded by 1:

    >>> r3 = geron_deep_rnn([[10.0]] * 5, weights=W, activation="tanh")
    >>> max(abs(v[0]) for v in r3["outputs"]) <= 1.0
    True

    The parameter count is exact: a 2 -> 3 layer holds 2*3 + 3*3 + 3 = 18.

    >>> geron_deep_rnn([[0.0, 0.0]], hidden_sizes=3)["n_params"]
    18

    References
    ----------
    Géron Ch 13
    """
    Xa = np.atleast_2d(np.asarray(X, dtype=float))
    if Xa.size == 0:
        raise ValueError("geron_deep_rnn: X is empty")
    if not np.all(np.isfinite(Xa)):
        raise ValueError("geron_deep_rnn: X contains non-finite values")
    if activation not in ("tanh", "relu"):
        raise ValueError(f"geron_deep_rnn: activation must be 'tanh' or 'relu', got {activation!r}")
    phi = np.tanh if activation == "tanh" else (lambda z: np.maximum(z, 0.0))
    T, d = Xa.shape

    if weights is not None:
        layers = []
        fan = d
        for i, trio in enumerate(weights):
            if len(trio) != 3:
                raise ValueError(f"geron_deep_rnn: layer {i} must be (Wx, Wh, b), got {len(trio)} items")
            Wx = np.atleast_2d(np.asarray(trio[0], dtype=float))
            Wh = np.atleast_2d(np.asarray(trio[1], dtype=float))
            b = np.atleast_1d(np.asarray(trio[2], dtype=float))
            if Wx.shape[0] != fan:
                raise ValueError(f"geron_deep_rnn: layer {i} Wx has {Wx.shape[0]} rows but its input width is {fan}")
            H = Wx.shape[1]
            if Wh.shape != (H, H):
                raise ValueError(f"geron_deep_rnn: layer {i} Wh must be ({H}, {H}), got {Wh.shape}")
            if b.size != H:
                raise ValueError(f"geron_deep_rnn: layer {i} b must have {H} entries, got {b.size}")
            layers.append((Wx, Wh, b))
            fan = H
        sizes = [Wx.shape[1] for Wx, _, _ in layers]
    else:
        if np.ndim(hidden_sizes) == 0:
            L = 1 if n_layers is None else int(n_layers)
            if L < 1:
                raise ValueError(f"geron_deep_rnn: n_layers must be >= 1, got {n_layers!r}")
            sizes = [int(hidden_sizes)] * L
        else:
            sizes = [int(h) for h in hidden_sizes]
            if n_layers is not None and int(n_layers) != len(sizes):
                raise ValueError(
                    f"geron_deep_rnn: n_layers={n_layers} contradicts {len(sizes)} entries in hidden_sizes"
                )
        if not sizes or any(h < 1 for h in sizes):
            raise ValueError(f"geron_deep_rnn: hidden sizes must all be >= 1, got {hidden_sizes!r}")
        s = int(seed) % 2**32

        def draw(shape, sd):
            nonlocal s
            n = int(np.prod(shape))
            u = np.empty(n)
            for i in range(n):
                s = (1664525 * s + 1013904223) % 2**32
                u[i] = (s + 0.5) / 2**32
            return ((2.0 * u - 1.0) * np.sqrt(3.0) * sd).reshape(shape)

        layers = []
        fan = d
        for H in sizes:
            layers.append((draw((fan, H), 1.0 / np.sqrt(fan)), draw((H, H), 1.0 / np.sqrt(H)), np.zeros(H)))
            fan = H

    L = len(layers)
    states = [[] for _ in range(L)]
    h_prev = [np.zeros(Wx.shape[1]) for Wx, _, _ in layers]
    outputs = []
    for t in range(T):
        inp = Xa[t]
        for l, (Wx, Wh, b) in enumerate(layers):
            h = phi(inp @ Wx + h_prev[l] @ Wh + b)
            h_prev[l] = h
            states[l].append(h.tolist())
            inp = h
        outputs.append(inp.tolist())

    n_params = int(sum(Wx.size + Wh.size + b.size for Wx, Wh, b in layers))
    norms = [[float(np.linalg.norm(h)) for h in layer] for layer in states]

    return RichResult(
        title="Deep (stacked) RNN",
        summary_lines=[("Layers", L), ("Time steps", T), ("Parameters", n_params)],
        interpretation="Depth stacks representations at each time step; recurrence carries state along time within a layer.",
        payload={
            "outputs": outputs,
            "states": states,
            "final_states": [h.tolist() for h in h_prev],
            "layer_sizes": [int(Wx.shape[1]) for Wx, _, _ in layers],
            "n_layers": int(L),
            "n_params": n_params,
            "state_norms": norms,
            "activation": activation,
            "estimate": float(np.mean(outputs)),
            "n": int(T),
            "method": "stacked RNN forward pass, bottom-up per time step",
        },
    )


def cheatsheet():
    return "hmdrnn: Deep (stacked) RNN: multiple recurrent layers"
