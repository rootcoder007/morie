# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SENet channel-wise Squeeze-and-Excitation attention."""

import numpy as np

from ._richresult import RichResult
from .grsig import geron_sigmoid

__all__ = ["geron_senet_squeeze_excite"]

_METHOD = "Squeeze-and-Excitation channel attention"


def geron_senet_squeeze_excite(X, W1, W2):
    r"""Recalibrate feature maps channel by channel.

    .. math::
        \mathbf{z} = \mathrm{GAP}(X), \qquad
        \mathbf{s} = \sigma\bigl(W_2\,\mathrm{ReLU}(W_1 \mathbf{z})\bigr),
        \qquad Y = \mathbf{s} \odot X

    Squeeze: global average pooling collapses each channel to one number,
    which is the only step that gives the block a *global* receptive
    field.  Excite: a two-layer bottleneck (``W1`` reduces by ratio ``r``,
    ``W2`` restores) learns which channels matter for this particular
    input.  Sigmoid, not softmax -- channels are not competing for a
    fixed budget, several can be on at once.  Gates land in ``(0, 1)`` so
    the block can only attenuate, never amplify, which is what keeps it
    safe to drop into an existing network.

    Parameters
    ----------
    X : array-like, shape (H, W, C)
    W1 : array-like, shape (C // r, C)
    W2 : array-like, shape (C, C // r)

    Returns
    -------
    RichResult
        Payload keys ``output``, ``scale`` (per channel), ``squeeze``,
        ``reduction_ratio``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 12, SENet section.

    Examples
    --------
    Two channels with means 1 and 3; ``W1 = W2 = I`` so the gates are
    ``sigma(1)`` and ``sigma(3)``:

    >>> X = [[[1.0, 3.0]]]
    >>> I = [[1.0, 0.0], [0.0, 1.0]]
    >>> r = geron_senet_squeeze_excite(X, I, I)
    >>> [round(v, 6) for v in r["squeeze"]]
    [1.0, 3.0]
    >>> [round(v, 6) for v in r["scale"]]
    [0.731059, 0.952574]
    >>> round(r["output"][0][0][1], 6)
    2.857722

    Gates never exceed 1, so SE only attenuates:

    >>> all(s < 1.0 for s in r["scale"])
    True
    """
    A = np.asarray(X, dtype=float)
    if A.ndim != 3 or A.size == 0:
        raise ValueError(f"X must be a non-empty (H, W, C) array, got shape {A.shape}.")
    if not np.all(np.isfinite(A)):
        raise ValueError("X contains non-finite values.")
    C = A.shape[2]
    A1 = np.atleast_2d(np.asarray(W1, dtype=float))
    A2 = np.atleast_2d(np.asarray(W2, dtype=float))
    if A1.shape[1] != C:
        raise ValueError(f"W1 must have {C} columns to consume the squeeze vector, got {A1.shape[1]}.")
    if A2.shape[1] != A1.shape[0]:
        raise ValueError(
            f"W2 must have {A1.shape[0]} columns to match the bottleneck, got {A2.shape[1]}."
        )
    if A2.shape[0] != C:
        raise ValueError(f"W2 must produce {C} channel gates, got {A2.shape[0]}.")
    if not np.all(np.isfinite(A1)) or not np.all(np.isfinite(A2)):
        raise ValueError("W1 and W2 must be finite.")

    z = A.mean(axis=(0, 1))
    hidden = np.maximum(0.0, A1 @ z)
    s = np.asarray(geron_sigmoid(A2 @ hidden)["sigma"], dtype=float).ravel()
    Y = A * s[None, None, :]

    return RichResult(
        title="Squeeze-and-Excitation",
        summary_lines=[("Channels", int(C)), ("Bottleneck", int(A1.shape[0]))],
        payload={
            "output": Y.tolist(),
            "scale": s.tolist(),
            "squeeze": z.tolist(),
            "hidden": hidden.tolist(),
            "reduction_ratio": float(C / A1.shape[0]),
            "estimate": Y.tolist(),
            "n": int(C),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grsen: z=GAP(X); s=sigmoid(W2 relu(W1 z)); Y = s*X per channel; sigmoid not softmax"
