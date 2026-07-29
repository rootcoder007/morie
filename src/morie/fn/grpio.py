# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Perceiver IO: a small latent array cross-attends to a large input."""

import numpy as np

from ._richresult import RichResult
from .grsdpa import attend

__all__ = ["geron_perceiver_io"]

_METHOD = "Perceiver IO cross-attention bottleneck"


def geron_perceiver_io(X, Z_latent, output_queries, n_iter=1):
    r"""Encode into a fixed-size latent, process, then decode by query.

    .. math::
        z &\leftarrow \mathrm{CrossAttn}(Q = Z,\; K = V = X)\\
        z &\leftarrow \mathrm{SelfAttn}(z)\quad \text{(repeat)}\\
        y &= \mathrm{CrossAttn}(Q = \text{output queries},\; K = V = z)

    The point is the asymmetry.  Self-attention over :math:`M` inputs
    costs :math:`O(M^2)`; here the only contact with the input is a
    cross-attention costing :math:`O(NM)` for a *fixed* latent size
    :math:`N \ll M`, and all the depth happens in :math:`O(N^2)` latent
    space.  So the cost is linear in input size, which is what lets one
    architecture eat images, audio and point clouds.  Decoding by
    arbitrary output queries -- the "IO" -- means the output shape is
    decoupled from both.

    This is the attention structure with identity projections; supply
    pre-projected arrays if you want learned ones (the kernel is
    :func:`morie.fn.grsdpa.attend`).

    Parameters
    ----------
    X : array-like, shape (M, d)
        Input array (the big one).
    Z_latent : array-like, shape (N, d)
        Learned latent array, ``N`` much smaller than ``M``.
    output_queries : array-like, shape (O, d)
    n_iter : int, optional
        Cross-attend/self-attend rounds, at least 1.

    Returns
    -------
    RichResult
        Payload keys ``output``, ``latent``, ``cross_weights``,
        ``latent_self_weights``, ``output_weights``, ``complexity_ratio``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 16, Perceiver / Perceiver IO section.

    Examples
    --------
    A single latent row averages nothing -- it attends over 4 inputs and
    a uniform key match makes it their mean:

    >>> X = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
    >>> r = geron_perceiver_io(X, [[0.0, 0.0]], [[0.0, 0.0]])
    >>> [round(v, 6) for v in r["latent"][0]]
    [2.5, 0.0]
    >>> [round(v, 6) for v in r["output"][0]]
    [2.5, 0.0]

    The latent size, not the input size, sets the working cost:

    >>> r["complexity_ratio"]
    4.0
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    Z = np.atleast_2d(np.asarray(Z_latent, dtype=float))
    O = np.atleast_2d(np.asarray(output_queries, dtype=float))
    for name, M in (("X", A), ("Z_latent", Z), ("output_queries", O)):
        if M.ndim != 2 or M.size == 0:
            raise ValueError(f"{name} must be a non-empty 2-D array, got shape {M.shape}.")
        if not np.all(np.isfinite(M)):
            raise ValueError(f"{name} contains non-finite values.")
    if Z.shape[1] != A.shape[1]:
        raise ValueError(
            f"Z_latent width {Z.shape[1]} != input width {A.shape[1]}; "
            "project them to a common d first."
        )
    if O.shape[1] != Z.shape[1]:
        raise ValueError(
            f"output_queries width {O.shape[1]} != latent width {Z.shape[1]}."
        )
    if Z.shape[0] > A.shape[0]:
        raise ValueError(
            f"the latent array ({Z.shape[0]} rows) is not smaller than the input "
            f"({A.shape[0]} rows); Perceiver's whole point is N << M."
        )
    n_iter = int(n_iter)
    if n_iter < 1:
        raise ValueError(f"n_iter must be at least 1, got {n_iter}.")

    z = Z
    cross_w, self_w = [], []
    for _ in range(n_iter):
        z, w = attend(z, A, A)
        cross_w.append(w.tolist())
        z, w2 = attend(z, z, z)
        self_w.append(w2.tolist())
    y, ow = attend(O, z, z)

    return RichResult(
        title="Perceiver IO",
        summary_lines=[("Inputs", int(A.shape[0])), ("Latents", int(Z.shape[0])),
                       ("Outputs", int(O.shape[0]))],
        payload={
            "output": y.tolist(),
            "latent": z.tolist(),
            "cross_weights": cross_w,
            "latent_self_weights": self_w,
            "output_weights": ow.tolist(),
            "complexity_ratio": float(A.shape[0] / Z.shape[0]),
            "estimate": y.tolist(),
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grpio: latent cross-attends to input O(NM), self-attends O(N^2), output queries decode"
