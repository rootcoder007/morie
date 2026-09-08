# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stratified sampling preserves class/strata proportions in each split."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_stratified_sampling"]


def geron_stratified_sampling(X, y=None, stratum=None, n_total=None, seed=0):
    """
    Stratified sampling preserves class/strata proportions in each split.

    Formula: n_h / n = N_h / N for each stratum h

    Proportional allocation cannot generally be met exactly in integers,
    so the allocation uses the largest-remainder (Hamilton) rule: give
    each stratum ``floor(n_total * N_h / N)``, then hand the leftover
    units to the strata with the largest fractional remainders (ties
    broken by the larger stratum). That guarantees the allocation sums to
    `n_total` exactly and no stratum is off by more than one unit -- the
    failure mode of naive rounding, which can miss the total entirely.

    Every stratum must get at least one unit; a request that would empty
    a stratum is an error, because the resulting sample is no longer
    stratified.

    Parameters
    ----------
    X : array-like
        Data rows (n, ...).
    y : array-like, optional
        Labels; used as the strata when `stratum` is None.
    stratum : array-like, optional
        Stratum key per row.
    n_total : int
        Sample size (1 <= n_total <= n). Required.
    seed : int, default 0
        LCG seed for the within-stratum draw.

    Returns
    -------
    result : RichResult
        Keys: indices, X_sample, y_sample, allocation, population_share,
        sample_share, max_share_error, estimate, n, method.

    Examples
    --------
    Four rows in stratum 0 and two in stratum 1, sample 3: the shares
    2/3 and 1/3 are reproduced exactly.

    >>> X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    >>> r = geron_stratified_sampling(X, [0, 0, 0, 0, 1, 1], n_total=3)
    >>> sorted(r["allocation"].items())
    [(0, 2), (1, 1)]
    >>> int(len(r["indices"]))
    3
    >>> round(float(r["max_share_error"]), 12)
    0.0

    Largest remainder in action: 5 and 3 rows, sample 4 -> exact quotas
    2.5 and 1.5, so the leftover unit goes to the larger stratum.

    >>> r2 = geron_stratified_sampling([[0.0]] * 8, [0] * 5 + [1] * 3, n_total=4)
    >>> sorted(r2["allocation"].items())
    [(0, 3), (1, 1)]

    References
    ----------
    Géron Ch 2
    """
    A = np.asarray(X)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.shape[0] == 0:
        raise ValueError("geron_stratified_sampling: X is empty")
    keys = stratum if stratum is not None else y
    if keys is None:
        raise ValueError("geron_stratified_sampling: supply strata via `stratum` or labels via `y`")
    k = np.asarray(keys).ravel()
    if k.size != A.shape[0]:
        raise ValueError(f"geron_stratified_sampling: X has {A.shape[0]} rows but the strata have {k.size} entries")
    if n_total is None:
        raise ValueError("geron_stratified_sampling: n_total is required")
    m = int(n_total)
    n = int(A.shape[0])
    if not (1 <= m <= n):
        raise ValueError(f"geron_stratified_sampling: n_total must lie in 1..{n}, got {m}")

    uniq, counts = np.unique(k, return_counts=True)
    if uniq.size > m:
        raise ValueError(
            f"geron_stratified_sampling: {uniq.size} strata cannot each receive a unit from a sample of {m}"
        )

    quota = m * counts / n
    base = np.floor(quota).astype(int)
    base = np.maximum(base, 1)
    left = m - int(base.sum())
    if left < 0:
        # Too many strata forced to 1; take back from the largest allocations.
        order = np.argsort(-(counts.astype(float)))
        i = 0
        while left < 0:
            j = order[i % order.size]
            if base[j] > 1:
                base[j] -= 1
                left += 1
            i += 1
    else:
        rema = quota - np.floor(quota)
        order = np.lexsort((-counts, -rema))
        for i in range(left):
            base[order[i % order.size]] += 1

    rng = int(seed) % 2**32

    def _u():
        nonlocal rng
        rng = (1664525 * rng + 1013904223) % 2**32
        return (rng + 0.5) / 2**32

    picked = []
    alloc = {}
    for h, want in zip(uniq, base):
        idx = np.flatnonzero(k == h).tolist()
        for i in range(len(idx) - 1, 0, -1):
            j = int(_u() * (i + 1))
            idx[i], idx[j] = idx[j], idx[i]
        picked.extend(sorted(idx[: int(want)]))
        alloc[int(h) if np.issubdtype(uniq.dtype, np.integer) else h] = int(want)
    picked = np.asarray(sorted(picked), dtype=int)

    pop_share = counts / n
    samp_share = base / m
    err = float(np.max(np.abs(pop_share - samp_share)))
    ys = np.asarray(y)[picked] if y is not None else None

    return RichResult(
        title="Stratified sample",
        summary_lines=[
            ("Population", n),
            ("Sample", int(picked.size)),
            ("Strata", int(uniq.size)),
            ("Max share error", err),
        ],
        interpretation=(
            "Stratifying removes sampling variance in the strata proportions entirely; with a rare "
            "class, a simple random split can miss it, and no amount of averaging fixes that."
        ),
        payload={
            "indices": picked,
            "X_sample": A[picked],
            "y_sample": ys,
            "allocation": alloc,
            "strata": uniq,
            "population_share": pop_share,
            "sample_share": samp_share,
            "max_share_error": err,
            "estimate": err,
            "n": int(picked.size),
            "method": "Proportional allocation with the largest-remainder rule, LCG draw within strata",
        },
    )


def cheatsheet():
    return "hmstr: Stratified sampling preserves class/strata proportions in each split"
