# morie.fn -- function file (rootcoder007/morie)
"""Balanced repeated replication half-samples."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["brr_balanced"]


def brr_balanced(strata, fay_k=0.0):
    r"""Construct a balanced set of half-sample replicate weights.

    For a design with :math:`H` strata of two PSUs each, BRR selects one PSU
    per stratum per replicate, following a **Hadamard matrix** so the
    selections are orthogonal across replicates. That orthogonality is the
    whole construction: it means :math:`R \approx H` replicates suffice for an
    unbiased variance estimate, where independent random half-samples would
    need far more.

    The replicate count is the next power of two at or above :math:`H` (minimum 4),
    because Hadamard matrices exist only at those orders. Using fewer
    replicates than strata, or an unbalanced set, silently biases the variance
    estimate.

    With a Fay adjustment the non-selected PSU is down-weighted to :math:`k`
    rather than dropped, which keeps every unit present in every replicate.
    That is what stops replicates failing on empty cells, and it must be paired
    with the :math:`(1-k)^2` divisor in
    :func:`~morie.fn.brrvar.brr_variance`.

    Parameters
    ----------
    strata : array-like
        Stratum label per unit. Each stratum must contain exactly 2 PSUs.
    fay_k : float
        Fay factor in [0, 1). 0 drops the non-selected PSU.

    Returns
    -------
    RichResult
        ``replicate_weights`` ``(R, n)``, ``n_replicates``, ``hadamard``,
        ``n_strata``, ``fay_k``.

    References
    ----------
    Wolter, K. M. (2007). *Introduction to Variance Estimation* (2nd ed.).
        Springer.

    Examples
    --------
    Replicate count is the next power of two at or above the stratum count,
    which is where Hadamard matrices exist.

    >>> import numpy as np
    >>> s = np.repeat(np.arange(5), 2)
    >>> r = brr_balanced(s)
    >>> int(r["n_strata"]), int(r["n_replicates"])
    (5, 8)

    Without a Fay adjustment each replicate keeps exactly one PSU per stratum,
    at double weight.

    >>> w = r["replicate_weights"][0]
    >>> sorted(set(np.round(w, 6).tolist()))
    [0.0, 2.0]

    A Fay adjustment keeps every unit present, which is what stops replicates
    failing on empty cells.

    >>> f = brr_balanced(s, fay_k=0.3)
    >>> bool(f["replicate_weights"].min() > 0)
    True

    >>> brr_balanced([0, 0, 1])
    Traceback (most recent call last):
        ...
    ValueError: stratum 1 has 1 PSUs; BRR requires exactly 2
    """
    s = np.asarray(strata).ravel()
    levels, inv = np.unique(s, return_inverse=True)
    H = levels.size
    for h, lv in enumerate(levels):
        cnt = int(np.sum(inv == h))
        if cnt != 2:
            raise ValueError(f"stratum {lv} has {cnt} PSUs; BRR requires exactly 2")
    fay_k = float(fay_k)
    if not 0.0 <= fay_k < 1.0:
        raise ValueError("fay_k must be in [0, 1)")

    # Full balance needs the COLUMNS of the R x H sign matrix to be
    # orthogonal. Slicing a Sylvester Hadamard to a non-power-of-two
    # row count destroys that (off-diagonal inner products of 4 at
    # H = 9..12), so R is the next power of two >= max(H, 4): a few
    # extra replicates, exact balance.
    R = 4
    while R < H:
        R *= 2
    R = max(R, 4)
    # Sylvester construction; valid for R a power of 2, padded up otherwise.
    size = 1
    while size < R:
        size *= 2
    Hm = np.ones((1, 1))
    while Hm.shape[0] < size:
        Hm = np.block([[Hm, Hm], [Hm, -Hm]])
    Hm = Hm[:R, :H] if Hm.shape[0] >= R else Hm

    n = s.size
    W = np.empty((R, n))
    for r in range(R):
        for h in range(H):
            members = np.flatnonzero(inv == h)
            pick = 0 if Hm[r, h] > 0 else 1
            W[r, members[pick]] = 2.0 - fay_k
            W[r, members[1 - pick]] = fay_k
    return RichResult(
        title="BRR half-samples",
        summary_lines=[("strata", int(H)), ("replicates", int(R)),
                       ("Fay k", fay_k)],
        warnings=["pair these weights with the (1-k)^2 divisor in brr_variance; "
                  "using fewer replicates than strata biases the variance"],
        payload={
            "replicate_weights": W, "n_replicates": int(R),
            "hadamard": Hm, "n_strata": int(H), "fay_k": fay_k,
            "n": int(n), "method": "brr_balanced",
        },
    )


def cheatsheet():
    return "brrest: Hadamard orthogonality is why R~H replicates suffice; pair with brr_variance's (1-k)^2"
