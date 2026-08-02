# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stacked autoencoder: a symmetric encoder/decoder tower."""

from . import _array_core as np

from ._richresult import RichResult
from .grsig import geron_sigmoid

__all__ = ["geron_stacked_autoencoder"]

_METHOD = "Stacked (deep) autoencoder forward pass"


def _act(z, kind):
    if kind == "relu":
        return np.maximum(0.0, z)
    if kind == "sigmoid":
        return np.asarray(geron_sigmoid(z)["sigma"], dtype=float).reshape(z.shape)
    if kind == "tanh":
        return np.tanh(z)
    if kind == "linear":
        return z
    raise ValueError(f"activation must be relu, sigmoid, tanh or linear, got {kind!r}.")


def geron_stacked_autoencoder(x, layer_weights, activation="relu", tied=True,
                              output_activation="linear"):
    r"""Push an input down to the code and back up again.

    .. math::
        x \to h_1 \to h_2 \to \dots \to h_L \to \dots \to \hat x

    ``layer_weights`` gives the *encoder* only; with ``tied=True`` the
    decoder reuses the transposes in reverse order.  Tying halves the
    parameter count and regularises -- it is the reason a deep
    autoencoder can be trained on a small dataset without immediately
    memorising it -- and it forces the architecture to be symmetric,
    which is the property this module exists to guarantee.  Pass
    ``tied=False`` with the decoder matrices appended to use untied
    weights.

    The narrowest layer is the code; if no layer is narrower than the
    input, the "bottleneck" is not one and the network can learn the
    identity, so that raises.

    Parameters
    ----------
    x : array-like, shape (m, n) or (n,)
    layer_weights : sequence of arrays
        Encoder matrices, each ``(in, out)``; with ``tied=False``, the
        decoder matrices follow in order.
    activation : {"relu", "sigmoid", "tanh", "linear"}, optional
    tied : bool, optional
    output_activation : str, optional
        Activation of the final reconstruction layer.

    Returns
    -------
    RichResult
        Payload keys ``reconstruction``, ``code``, ``activations``,
        ``reconstruction_error``, ``compression``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 18, Stacked Autoencoders section.

    Examples
    --------
    Two inputs into one code unit and back, tied weights: the code is
    ``x @ W = 3``, and the reconstruction is ``3 * W^T``.

    >>> W = [[1.0], [1.0]]
    >>> r = geron_stacked_autoencoder([[1.0, 2.0]], [W], activation="linear")
    >>> r["code"]
    [[3.0]]
    >>> r["reconstruction"]
    [[3.0, 3.0]]
    >>> r["compression"]
    2.0

    A perfect autoencoder has zero error; this one does not:

    >>> round(r["reconstruction_error"], 6)
    2.5
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.size == 0:
        raise ValueError("x is empty.")
    if not np.all(np.isfinite(X)):
        raise ValueError("x contains non-finite values.")
    mats = [np.atleast_2d(np.asarray(W, dtype=float)) for W in layer_weights]
    if not mats:
        raise ValueError("layer_weights is empty; an autoencoder needs at least one layer.")
    for i, W in enumerate(mats):
        if not np.all(np.isfinite(W)):
            raise ValueError(f"layer_weights[{i}] contains non-finite values.")

    if tied:
        enc = mats
        dec = [W.T for W in reversed(mats)]
    else:
        if len(mats) % 2 != 0:
            raise ValueError(
                f"untied weights need an encoder and a decoder matrix per level, "
                f"got {len(mats)} matrices."
            )
        half = len(mats) // 2
        enc, dec = mats[:half], mats[half:]

    width = X.shape[1]
    for i, W in enumerate(enc):
        if W.shape[0] != width:
            raise ValueError(
                f"encoder layer {i} expects {W.shape[0]} inputs but receives {width}."
            )
        width = W.shape[1]
    code_width = width
    if code_width >= X.shape[1]:
        raise ValueError(
            f"the code is {code_width} wide for a {X.shape[1]}-wide input; without a "
            "bottleneck the network can learn the identity."
        )
    for i, W in enumerate(dec):
        if W.shape[0] != width:
            raise ValueError(
                f"decoder layer {i} expects {W.shape[0]} inputs but receives {width}."
            )
        width = W.shape[1]
    if width != X.shape[1]:
        raise ValueError(
            f"the decoder ends at width {width} but the input is {X.shape[1]} wide; "
            "the tower is not symmetric."
        )

    acts = [X]
    h = X
    for W in enc:
        h = _act(h @ W, activation)
        acts.append(h)
    code = h
    for i, W in enumerate(dec):
        kind = output_activation if i == len(dec) - 1 else activation
        h = _act(h @ W, kind)
        acts.append(h)

    err = float(np.mean((h - X) ** 2))
    return RichResult(
        title="Stacked autoencoder",
        summary_lines=[("Code width", int(code.shape[1])), ("Reconstruction MSE", err)],
        payload={
            "reconstruction": h.tolist(),
            "code": code.tolist(),
            "activations": [a.tolist() for a in acts],
            "reconstruction_error": err,
            "compression": float(X.shape[1] / code.shape[1]),
            "tied": bool(tied),
            "estimate": h.tolist(),
            "n": int(X.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grstae: encoder matrices in, tied transposes back out; bottleneck enforced, symmetry checked"
