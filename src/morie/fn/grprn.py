# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Magnitude-based unstructured weight pruning."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_weight_pruning"]

_METHOD = "Magnitude-based unstructured pruning"


def geron_weight_pruning(W, sparsity):
    r"""Zero the smallest weights by magnitude.

    .. math::
        \mathrm{mask}_{ij} = \begin{cases}
            1 & |W_{ij}| > \tau\\ 0 & \text{otherwise}\end{cases},
        \qquad W \leftarrow W \odot \mathrm{mask}

    The threshold :math:`\tau` is chosen as the ``sparsity`` quantile of
    :math:`|W|`, so the requested fraction is hit exactly rather than
    approximately -- a fixed numeric threshold gives you whatever
    sparsity it happens to give.  Unstructured means the surviving
    weights are scattered, which shrinks the model on disk but not
    necessarily its runtime: dense kernels do not get faster from zeros.
    That trade is the reason structured pruning exists.

    Parameters
    ----------
    W : array-like
        Weight tensor.
    sparsity : float
        Fraction to zero, in ``[0, 1)``.

    Returns
    -------
    RichResult
        Payload keys ``W_pruned``, ``mask``, ``threshold``,
        ``achieved_sparsity``, ``n_pruned``, ``norm_retained``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 17, Weight Pruning section.

    Examples
    --------
    Half of four weights: the two smallest magnitudes go.

    >>> r = geron_weight_pruning([[1.0, -0.1], [0.05, 2.0]], sparsity=0.5)
    >>> r["W_pruned"]
    [[1.0, 0.0], [0.0, 2.0]]
    >>> r["achieved_sparsity"]
    0.5

    Magnitude, not sign -- a large negative weight survives:

    >>> geron_weight_pruning([-5.0, 0.1], sparsity=0.5)["W_pruned"]
    [-5.0, 0.0]
    """
    A = np.asarray(W, dtype=float)
    if A.size == 0:
        raise ValueError("W is empty.")
    if not np.all(np.isfinite(A)):
        raise ValueError("W contains non-finite values.")
    sparsity = float(sparsity)
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must lie in [0, 1), got {sparsity}.")

    mag = np.abs(A).ravel()
    k = int(round(sparsity * mag.size))
    if k == 0:
        thr = -np.inf
        mask = np.ones_like(A)
    else:
        order = np.argsort(mag, kind="mergesort")
        keep = np.ones(mag.size, dtype=float)
        keep[order[:k]] = 0.0
        mask = keep.reshape(A.shape)
        thr = float(mag[order[k - 1]])
    # np.where, not A * mask: multiplying a negative weight by 0.0 gives -0.0.
    P = np.where(mask > 0, A, 0.0)
    achieved = float(np.mean(mask == 0))

    return RichResult(
        title="Magnitude pruning",
        summary_lines=[("Requested sparsity", sparsity), ("Achieved", achieved),
                       ("Threshold", thr)],
        payload={
            "W_pruned": P.tolist(),
            "mask": mask.tolist(),
            "threshold": thr,
            "achieved_sparsity": achieved,
            "n_pruned": int(k),
            "norm_retained": float(np.linalg.norm(P) / np.linalg.norm(A)) if np.linalg.norm(A) > 0 else 0.0,
            "estimate": P.tolist(),
            "n": int(A.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grprn: zero the smallest |W| up to the requested fraction (quantile threshold, exact count)"
