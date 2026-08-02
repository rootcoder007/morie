# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DataLoader-style mini-batch index generation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_dataloader_minibatch"]

_METHOD = "Mini-batch index generation (DataLoader)"


def _lcg_permutation(n, seed):
    """Fisher-Yates shuffle driven by the reference LCG.

    ``s = (1664525 s + 1013904223) mod 2**32``, ``u = (s + 0.5)/2**32``.
    Every index appears exactly once -- a permutation, not sampling with
    replacement, which is what makes an epoch cover the whole dataset.
    """
    s = int(seed) % 2**32
    perm = np.arange(n)
    for i in range(n - 1, 0, -1):
        s = (1664525 * s + 1013904223) % 2**32
        u = (s + 0.5) / 2**32
        j = min(int(u * (i + 1)), i)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def geron_dataloader_minibatch(n, b, shuffle=True, seed=0, drop_last=False):
    r"""Split one epoch into mini-batches.

    .. math::
        \text{perm} = \mathrm{permute}(n),\qquad
        \text{batch}_i = \text{perm}[ib : (i+1)b]

    Shuffling matters because mini-batch gradient descent assumes each
    batch is an unbiased sample of the data.  Feed it a file sorted by
    label and every batch is a single class, so the gradient points
    somewhere no full-batch gradient ever would.

    The last batch is short when ``b`` does not divide ``n``.  Keeping
    it (the default) covers the whole epoch; dropping it keeps every
    batch the same size, which matters when the batch size is baked into
    a normalisation.

    Parameters
    ----------
    n : int
        Dataset size, at least 1.
    b : int
        Batch size, ``1 <= b <= n``.
    shuffle : bool, optional
        Default True.
    seed : int, optional
    drop_last : bool, optional
        Discard a final short batch. Default False.

    Returns
    -------
    RichResult
        Payload keys ``batches`` (list of index lists),
        ``n_batches``, ``batch_sizes``, ``permutation``,
        ``covers_all`` (True when every index appears exactly once),
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 10, DataLoader mini-batch GD section.

    Examples
    --------
    Unshuffled, the batches are simply consecutive slices:

    >>> r = geron_dataloader_minibatch(5, 2, shuffle=False)
    >>> r["batches"]
    [[0, 1], [2, 3], [4]]
    >>> r["batch_sizes"]
    [2, 2, 1]

    Dropping the short tail loses the last instance:

    >>> geron_dataloader_minibatch(5, 2, shuffle=False, drop_last=True)["batches"]
    [[0, 1], [2, 3]]

    Shuffled, the order changes but the epoch is still a permutation --
    every index exactly once:

    >>> r2 = geron_dataloader_minibatch(6, 3, shuffle=True, seed=1)
    >>> sorted(i for batch in r2["batches"] for i in batch)
    [0, 1, 2, 3, 4, 5]
    >>> r2["covers_all"]
    True
    """
    n = int(n)
    b = int(b)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}.")
    if not (1 <= b <= n):
        raise ValueError(f"b must lie in [1, {n}], got {b}.")

    perm = _lcg_permutation(n, seed) if shuffle else np.arange(n)
    batches = [perm[i:i + b].tolist() for i in range(0, n, b)]
    if drop_last and batches and len(batches[-1]) < b:
        batches = batches[:-1]
    if not batches:
        raise ValueError(
            f"drop_last discarded every batch: n = {n} is smaller than b = {b}."
        )
    seen = sorted(i for batch in batches for i in batch)
    covers = seen == list(range(n))

    return RichResult(
        title="Mini-batch loader",
        summary_lines=[("Batches", len(batches)), ("Batch size", b),
                       ("Shuffled", bool(shuffle))],
        payload={
            "batches": batches,
            "n_batches": len(batches),
            "batch_sizes": [len(x) for x in batches],
            "permutation": perm.tolist(),
            "covers_all": bool(covers),
            "drop_last": bool(drop_last),
            "seed": int(seed),
            "estimate": batches,
            "n": int(n),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grdlm: shuffle once per epoch (LCG Fisher-Yates), then slice into batches of b"
