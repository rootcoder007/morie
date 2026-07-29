# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sparse autoencoder: reconstruction error plus an L1 penalty on the code."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_sparse_autoencoder"]

_METHOD = "Sparse autoencoder objective (MSE + L1 on the code)"


def geron_sparse_autoencoder(x, hidden, decoded, lam=1e-3):
    r"""Score a sparse autoencoder's forward pass.

    .. math::
        L = \|x - \mathrm{Dec}(\mathrm{Enc}(x))\|^2 + \lambda\|h\|_1,
        \qquad h = \mathrm{Enc}(x)

    The L1 term is what stops an over-complete code from learning the
    identity: with more hidden units than inputs, perfect reconstruction
    is free, and only the sparsity pressure forces each unit to
    specialise.  L1, not L2 -- the absolute value has a constant
    gradient towards zero, so it drives small activations exactly to
    zero, where a squared penalty would only shrink them.  The reported
    activation sparsity is therefore the number to watch, not the loss.

    Parameters
    ----------
    x : array-like, shape (m, n) or (n,)
        Inputs.
    hidden : array-like, shape (m, k)
        Encoder activations.
    decoded : array-like, same shape as ``x``.
    lam : float, optional
        Non-negative sparsity weight.

    Returns
    -------
    RichResult
        Payload keys ``loss``, ``reconstruction_loss``, ``l1_penalty``,
        ``sparsity`` (fraction of near-zero activations),
        ``mean_activation``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 18, Sparse Autoencoders section.

    Examples
    --------
    Perfect reconstruction, code ``[0, 2]``: the loss is all penalty,
    ``lam * 2``.

    >>> r = geron_sparse_autoencoder([[1.0]], [[0.0, 2.0]], [[1.0]], lam=0.5)
    >>> r["reconstruction_loss"], r["l1_penalty"]
    (0.0, 1.0)
    >>> r["loss"]
    1.0
    >>> r["sparsity"]
    0.5

    A reconstruction error of 1 on one input costs 1:

    >>> geron_sparse_autoencoder([[1.0]], [[0.0]], [[0.0]], lam=0.0)["loss"]
    1.0
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    H = np.atleast_2d(np.asarray(hidden, dtype=float))
    D = np.atleast_2d(np.asarray(decoded, dtype=float))
    if X.size == 0:
        raise ValueError("x is empty.")
    if D.shape != X.shape:
        raise ValueError(f"decoded has shape {D.shape} but x has {X.shape}.")
    if H.shape[0] != X.shape[0]:
        raise ValueError(f"hidden has {H.shape[0]} rows but x has {X.shape[0]}.")
    for name, M in (("x", X), ("hidden", H), ("decoded", D)):
        if not np.all(np.isfinite(M)):
            raise ValueError(f"{name} contains non-finite values.")
    lam = float(lam)
    if not np.isfinite(lam) or lam < 0:
        raise ValueError(f"lam must be finite and non-negative, got {lam}.")

    recon = float(np.sum((X - D) ** 2))
    l1 = float(lam * np.sum(np.abs(H)))
    sparsity = float(np.mean(np.abs(H) < 1e-8))

    return RichResult(
        title="Sparse autoencoder",
        summary_lines=[("Loss", recon + l1), ("Reconstruction", recon),
                       ("L1 penalty", l1), ("Sparsity", sparsity)],
        payload={
            "loss": recon + l1,
            "reconstruction_loss": recon,
            "l1_penalty": l1,
            "sparsity": sparsity,
            "mean_activation": float(np.mean(np.abs(H))),
            "code_size": int(H.shape[1]),
            "estimate": recon + l1,
            "n": int(X.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grsae: L = ||x - dec||^2 + lam ||h||_1; L1 zeroes units outright, L2 would only shrink them"
