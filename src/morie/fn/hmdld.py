# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DataLoader for mini-batch iteration with shuffling and parallel workers."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_dataloader"]


def _lcg(n, seed):
    """n deterministic uniforms in [0, 1) from the standard LCG."""
    s = int(seed) % 2**32
    u = np.empty(n)
    for i in range(n):
        s = (1664525 * s + 1013904223) % 2**32
        u[i] = (s + 0.5) / 2**32
    return u


def geron_dataloader(dataset, batch_size, shuffle=False, drop_last=False, seed=0, num_workers=0):
    """
    DataLoader for mini-batch iteration with shuffling and parallel workers.

    Formula: for x_b, y_b in DataLoader(dataset, batch_size, shuffle)

    The index plan is materialised rather than hidden behind an iterator,
    so the epoch is inspectable: ``batches`` holds the actual indices of
    each mini-batch, in order. Shuffling is a Fisher-Yates permutation
    driven by a deterministic LCG, which makes an epoch reproducible from
    ``seed`` alone and guarantees every index appears exactly once -- the
    property "sample with replacement" quietly breaks.

    ``drop_last`` discards the short final batch, which is what you want
    when a layer's statistics depend on a fixed batch size (batch norm).

    ``num_workers`` only affects the reported worker assignment; batch
    contents are worker-count independent by construction, which is the
    invariant that makes a parallel loader safe.

    Parameters
    ----------
    dataset : array-like or int
        The data (batched along the first axis), or just its length.
    batch_size : int
        Positive.
    shuffle : bool, default False
    drop_last : bool, default False
    seed : int, default 0
    num_workers : int, default 0
        Non-negative; 0 means load in the main process.

    Returns
    -------
    result : RichResult
        Keys: batches, order, n_batches, last_batch_size, dropped,
        worker_assignment, batch_data, estimate, n, method.

    Examples
    --------
    Seven items in batches of three, unshuffled:

    >>> r = geron_dataloader(7, batch_size=3)
    >>> r["batches"]
    [[0, 1, 2], [3, 4, 5], [6]]
    >>> r["n_batches"], r["last_batch_size"], r["dropped"]
    (3, 1, 0)

    ``drop_last`` throws the short batch away:

    >>> r2 = geron_dataloader(7, 3, drop_last=True)
    >>> r2["batches"]
    [[0, 1, 2], [3, 4, 5]]
    >>> r2["dropped"]
    1

    Shuffling is a permutation -- every index appears exactly once:

    >>> r3 = geron_dataloader(7, 3, shuffle=True, seed=42)
    >>> sorted(r3["order"])
    [0, 1, 2, 3, 4, 5, 6]
    >>> r3["order"] == list(range(7))
    False

    Real data comes back sliced:

    >>> r4 = geron_dataloader([[1.0], [2.0], [3.0]], batch_size=2)
    >>> r4["batch_data"][0]
    [[1.0], [2.0]]

    References
    ----------
    Géron Ch 10
    """
    bs = int(batch_size)
    if bs < 1:
        raise ValueError(f"geron_dataloader: batch_size must be >= 1, got {batch_size!r}")
    nw = int(num_workers)
    if nw < 0:
        raise ValueError(f"geron_dataloader: num_workers must be non-negative, got {num_workers!r}")

    data = None
    if isinstance(dataset, (int, np.integer)):
        m = int(dataset)
    else:
        data = np.asarray(dataset)
        m = int(data.shape[0]) if data.ndim else 0
    if m < 1:
        raise ValueError(f"geron_dataloader: dataset must contain at least one item, got length {m}")

    order = np.arange(m)
    if shuffle:
        u = _lcg(m - 1, seed)
        for i in range(m - 1, 0, -1):
            j = int(u[m - 1 - i] * (i + 1))
            j = min(j, i)
            order[i], order[j] = order[j], order[i]

    n_full = m // bs
    batches = [order[i * bs : (i + 1) * bs].tolist() for i in range(n_full)]
    rest = order[n_full * bs :].tolist()
    dropped = 0
    if rest:
        if drop_last:
            dropped = len(rest)
        else:
            batches.append(rest)

    batch_data = None
    if data is not None:
        batch_data = [data[np.asarray(b)].tolist() for b in batches]

    assign = [i % nw for i in range(len(batches))] if nw > 0 else [0] * len(batches)

    return RichResult(
        title="DataLoader epoch plan",
        summary_lines=[("Batches", len(batches)), ("Batch size", bs), ("Dropped", dropped)],
        interpretation="Shuffling is a permutation: each item is visited exactly once per epoch.",
        payload={
            "batches": batches,
            "order": order.tolist(),
            "n_batches": int(len(batches)),
            "last_batch_size": int(len(batches[-1])) if batches else 0,
            "dropped": int(dropped),
            "batch_size": bs,
            "shuffle": bool(shuffle),
            "drop_last": bool(drop_last),
            "worker_assignment": assign,
            "num_workers": nw,
            "batch_data": batch_data,
            "estimate": float(len(batches)),
            "n": int(m),
            "method": "mini-batch index plan with deterministic Fisher-Yates shuffling",
        },
    )


def cheatsheet():
    return "hmdld: DataLoader for mini-batch iteration with shuffling and parallel workers"
