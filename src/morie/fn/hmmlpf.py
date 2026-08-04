# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multilayer perceptron forward pass."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_mlp"]

_METHOD = "MLP forward pass"


def _sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


_ACTIVATIONS = {
    "relu": lambda z: np.maximum(z, 0.0),
    "tanh": np.tanh,
    "sigmoid": _sigmoid,
    "softmax": _softmax,
    "identity": lambda z: z,
    "linear": lambda z: z,
}


def geron_mlp(X, weights, biases, activations):
    """
    Multilayer perceptron forward pass.

    Formula: a^(l+1) = phi(W^(l+1) a^(l) + b^(l+1))

    Weights are stored as ``(fan_in, fan_out)`` and applied on the right
    (``a @ W + b``), so a batch of rows flows through without any
    transposing.  Every layer's pre-activation and activation are
    returned: those are exactly the quantities backpropagation needs, and
    a forward pass that discards them forces a second pass later.

    A network of all-linear activations is refused-by-warning rather than
    silently accepted, because stacking linear layers collapses to a
    single linear map -- depth buys nothing without a non-linearity.

    Parameters
    ----------
    X : array-like, shape (m, n_in) or (n_in,)
        Input batch.
    weights : sequence of array-like
        ``weights[l]`` has shape ``(fan_in_l, fan_out_l)`` and
        ``fan_in_{l+1} == fan_out_l``.
    biases : sequence of array-like
        ``biases[l]`` has length ``fan_out_l``.
    activations : sequence of str or callable
        One per layer: "relu", "tanh", "sigmoid", "softmax",
        "identity", or any callable applied elementwise/rowwise.

    Returns
    -------
    result : RichResult
        Keys: output, activations, pre_activations, n_parameters,
        estimate, n, method.

    Examples
    --------
    A two-layer net computing XOR exactly, with the textbook
    step-like ReLU construction:

    >>> W1 = [[1.0, 1.0], [1.0, 1.0]]
    >>> b1 = [0.0, -1.0]
    >>> W2 = [[1.0], [-2.0]]
    >>> b2 = [0.0]
    >>> X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    >>> r = geron_mlp(X, [W1, W2], [b1, b2], ["relu", "identity"])
    >>> [float(v) for v in r["output"].ravel()]
    [0.0, 1.0, 1.0, 0.0]

    The parameter count is ``2*2 + 2 + 2*1 + 1 = 9``:

    >>> r["n_parameters"]
    9

    Softmax output rows are distributions:

    >>> s = geron_mlp([[1.0]], [[[1.0, 2.0, 3.0]]], [[0.0, 0.0, 0.0]], ["softmax"])
    >>> round(float(np.sum(s["output"])), 12)
    1.0
    >>> [round(float(v), 6) for v in s["output"].ravel()]
    [0.090031, 0.244728, 0.665241]

    Shape mismatches are named, not broadcast:

    >>> geron_mlp([[1.0]], [[[1.0, 2.0]]], [[0.0]], ["relu"])
    Traceback (most recent call last):
        ...
    ValueError: geron_mlp: biases[0] has 1 entries but weights[0] has 2 output units

    References
    ----------
    Géron Ch 9
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_mlp: X must be a non-empty 2-D array, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_mlp: X contains non-finite values")

    try:
        L = len(weights)
    except TypeError:
        raise ValueError("geron_mlp: weights must be a sequence of layer matrices") from None
    if L == 0:
        raise ValueError("geron_mlp: weights is empty; an MLP needs at least one layer")
    if len(biases) != L:
        raise ValueError(f"geron_mlp: {L} weight matrices but {len(biases)} bias vectors")
    if len(activations) != L:
        raise ValueError(f"geron_mlp: {L} weight matrices but {len(activations)} activations")

    a = A
    acts = []
    pres = []
    n_params = 0
    fns = []
    for l in range(L):
        W = np.atleast_2d(np.asarray(weights[l], dtype=float))
        b = np.atleast_1d(np.asarray(biases[l], dtype=float)).ravel()
        if W.shape[0] != a.shape[1]:
            raise ValueError(
                f"geron_mlp: weights[{l}] expects {W.shape[0]} inputs but the previous layer produced {a.shape[1]}"
            )
        if b.size != W.shape[1]:
            raise ValueError(f"geron_mlp: biases[{l}] has {b.size} entries but weights[{l}] has {W.shape[1]} output units")
        if not np.all(np.isfinite(W)) or not np.all(np.isfinite(b)):
            raise ValueError(f"geron_mlp: weights[{l}] or biases[{l}] contains non-finite values")
        spec = activations[l]
        if callable(spec):
            fn = spec
            fns.append(getattr(spec, "__name__", "callable"))
        else:
            key = str(spec).lower()
            if key not in _ACTIVATIONS:
                raise ValueError(
                    f"geron_mlp: activations[{l}] = {spec!r} is not one of {sorted(_ACTIVATIONS)} or a callable"
                )
            fn = _ACTIVATIONS[key]
            fns.append(key)
        z = a @ W + b
        a = np.asarray(fn(z), dtype=float)
        if a.shape != z.shape:
            raise ValueError(
                f"geron_mlp: activations[{l}] changed the shape from {z.shape} to {a.shape}"
            )
        pres.append(z)
        acts.append(a)
        n_params += int(W.size + b.size)

    warns = []
    if all(f in ("identity", "linear") for f in fns) and L > 1:
        warns.append(
            f"all {L} layers are linear, so the network collapses to a single linear map; depth adds nothing."
        )

    return RichResult(
        title="MLP forward pass",
        summary_lines=[
            ("Layers", L),
            ("Parameters", n_params),
            ("Output shape", str(a.shape)),
        ],
        warnings=warns,
        interpretation="Pre-activations and activations are kept for every layer: backpropagation needs both.",
        payload={
            "output": a,
            "activations": acts,
            "pre_activations": pres,
            "n_parameters": n_params,
            "layer_activations": fns,
            "estimate": float(np.mean(a)),
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmmlpf: MLP forward pass a <- phi(a W + b), keeping every pre-activation for backprop"


# compact alias per ledger/NAMING.md
geronmlp = geron_mlp
