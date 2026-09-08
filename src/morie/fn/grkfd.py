# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""K-fold cross-validation index generation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_kfold_cv"]

_METHOD = "K-fold cross-validation splits"


def _lcg_permutation(n, seed):
    """Fisher-Yates shuffle from the reference LCG.

    ``s = (1664525 s + 1013904223) mod 2**32``, ``u = (s + 0.5)/2**32``.
    """
    s = int(seed) % 2**32
    perm = np.arange(n)
    for i in range(n - 1, 0, -1):
        s = (1664525 * s + 1013904223) % 2**32
        u = (s + 0.5) / 2**32
        j = min(int(u * (i + 1)), i)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def geron_kfold_cv(n, K, shuffle=False, seed=0):
    r"""Partition ``range(n)`` into ``K`` folds and yield train/val pairs.

    Each instance is used for validation exactly once and for training
    exactly ``K - 1`` times.  That is the property the whole method
    rests on -- it is what makes the ``K`` scores an estimate of
    generalisation on *all* the data rather than on one lucky split --
    and it is verified here (``each_used_once``) rather than assumed.

    When ``K`` does not divide ``n`` the first ``n mod K`` folds get one
    extra instance, so fold sizes differ by at most 1.

    Parameters
    ----------
    n : int
        Dataset size.
    K : int
        Folds, ``2 <= K <= n``.
    shuffle : bool, optional
        Shuffle before splitting. Default False -- leave it False only
        when the data order is already arbitrary.
    seed : int, optional

    Returns
    -------
    RichResult
        Payload keys ``splits`` (list of ``(train_idx, val_idx)``),
        ``val_folds``, ``fold_sizes``, ``each_used_once``,
        ``train_size``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 2, K-fold Cross-Validation section.  Score the splits with
    :func:`morie.fn.grcvs.geron_cross_validation_score`.

    Examples
    --------
    Six instances into three folds, unshuffled:

    >>> r = geron_kfold_cv(6, 3)
    >>> r["val_folds"]
    [[0, 1], [2, 3], [4, 5]]
    >>> r["splits"][0][0]
    [2, 3, 4, 5]
    >>> r["each_used_once"]
    True

    Five into two: the remainder goes to the first fold, never dropped:

    >>> geron_kfold_cv(5, 2)["fold_sizes"]
    [3, 2]

    Leave-one-out is just ``K = n``:

    >>> geron_kfold_cv(4, 4)["fold_sizes"]
    [1, 1, 1, 1]
    """
    n = int(n)
    K = int(K)
    if n < 2:
        raise ValueError(f"n must be at least 2 to split at all, got {n}.")
    if not (2 <= K <= n):
        raise ValueError(
            f"K must lie in [2, {n}]; K = 1 leaves no validation fold and K > n "
            f"would create empty folds. Got {K}."
        )

    order = _lcg_permutation(n, seed) if shuffle else np.arange(n)
    folds = [f.tolist() for f in np.array_split(order, K)]
    splits = []
    for k in range(K):
        val = folds[k]
        train = [i for j, f in enumerate(folds) if j != k for i in f]
        splits.append((train, val))

    used = sorted(i for f in folds for i in f)
    each_once = used == list(range(n))

    return RichResult(
        title="K-fold cross-validation",
        summary_lines=[("K", K), ("Fold sizes", [len(f) for f in folds]),
                       ("Shuffled", bool(shuffle))],
        payload={
            "splits": splits,
            "val_folds": folds,
            "fold_sizes": [len(f) for f in folds],
            "each_used_once": bool(each_once),
            "train_size": [len(s[0]) for s in splits],
            "K": K,
            "seed": int(seed),
            "estimate": splits,
            "n": int(n),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grkfd: K folds, every instance validated exactly once; remainder spread over the first folds"


# compact alias per ledger/NAMING.md
geronkfoldcv = geron_kfold_cv
