# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shannon entropy at a tree node."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_shannon_entropy"]

_METHOD = "Shannon entropy (Eq 5-3)"


def geron_shannon_entropy(y, base=2.0):
    r"""Géron Eq 5-3.

    .. math::
        H_i = -\sum_{k:\, p_{i,k} > 0} p_{i,k} \log_2 p_{i,k}

    The sum skips empty classes, which is not a numerical dodge:
    :math:`p \log p \to 0` as :math:`p \to 0`, so the omitted terms are
    genuinely zero.  Computing them anyway is what produces the
    ``0 * -inf = nan`` that plagues naive implementations.

    In base 2 the unit is bits, and the maximum for ``K`` equally
    likely classes is exactly :math:`\log_2 K` -- 1 bit for a fair coin.

    Parameters
    ----------
    y : array-like
        Class labels at the node.
    base : float, optional
        Logarithm base, default 2 (bits). Use ``math.e`` for nats.

    Returns
    -------
    RichResult
        Payload keys ``entropy``, ``proportions``, ``classes``,
        ``max_possible``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 5, Eq 5-3 (Entropy).  ``grgin`` is the Gini alternative;
    ``grig`` builds the split criterion on top of this.

    Examples
    --------
    A fair coin is exactly one bit:

    >>> geron_shannon_entropy([0, 1])["entropy"]
    1.0

    Géron's 49/5 node: ``-(49/54)log2(49/54) - (5/54)log2(5/54)``:

    >>> round(geron_shannon_entropy([0] * 49 + [1] * 5)["entropy"], 6)
    0.445065

    A pure node has no uncertainty, and the empty-class term does not
    poison it:

    >>> geron_shannon_entropy([7, 7, 7, 7])["entropy"]
    0.0
    """
    y = np.asarray(y).ravel()
    if y.size == 0:
        raise ValueError("y is empty; entropy of an empty node is undefined.")
    base = float(base)
    if not np.isfinite(base) or base <= 1.0:
        raise ValueError(f"base must be a finite float greater than 1, got {base}.")

    classes, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    nz = p[p > 0]
    H = float(-np.sum(nz * (np.log(nz) / np.log(base))))
    H = H + 0.0 if H != 0.0 else 0.0   # a pure node gives -0.0; report plain 0.0
    K = int(classes.size)

    return RichResult(
        title="Shannon entropy",
        summary_lines=[("Entropy", H), ("Classes", K), ("n", int(y.size))],
        payload={
            "entropy": H,
            "proportions": p.tolist(),
            "classes": classes.tolist(),
            "counts": counts.tolist(),
            "max_possible": float(np.log(K) / np.log(base)),
            "base": base,
            "estimate": H,
            "n": int(y.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grent: H = -sum_{p>0} p log2 p, empty classes skipped -- Geron Eq 5-3"
