# morie.fn -- function file (rootcoder007/morie)
"""Differentially private k-means."""

from __future__ import annotations

import numpy as np

from ._dp import check_budget, clip_to_range
from ._richresult import RichResult

__all__ = ["dp_kmeans"]


def dp_kmeans(X, k=3, epsilon=1.0, n_iter=5, bounds=None, seed=None):
    r"""Lloyd's algorithm with every sufficient statistic released privately.

    Each iteration needs two things per cluster -- the sum of assigned points
    and their count -- and both are privatised by the Laplace mechanism. The
    centroid is their ratio, so the released centroids are post-processing and
    cost nothing beyond the statistics themselves.

    The budget must be split across **iterations as well as clusters**, since
    each pass is a fresh query against the same data. With ``n_iter`` passes
    each iteration gets :math:`\varepsilon/(2\,\text{n\_iter})` -- half for
    sums, half for counts. Running more iterations therefore makes the answer
    *worse*, not better, which inverts the usual intuition about convergence:
    with a fixed budget there is an optimal number of passes and it is small.

    Empty or nearly-empty clusters are the failure mode. A noisy count can go
    to zero or negative, and a centroid computed from it is meaningless; such
    clusters are reinitialised and counted in ``n_reinitialised``.

    Parameters
    ----------
    X : array-like
        Data ``(n, p)``.
    k : int
        Number of clusters.
    epsilon : float
        Total budget across all iterations.
    n_iter : int
        Lloyd iterations. More is not better here.
    bounds : tuple, optional
        ``(low, high)`` clipping bounds chosen independently of the data.
    seed : int, optional
        Seed; leave ``None`` for a real release.

    Returns
    -------
    RichResult
        ``centers``, ``labels``, ``epsilon_per_iteration``,
        ``n_reinitialised``, ``inertia``.

    References
    ----------
    Blum, A., Dwork, C., McSherry, F., & Nissim, K. (2005). Practical privacy:
        the SuLQ framework. *PODS 2005*, 128-138.
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. *FnT-TCS*, 9(3-4), 211-487.

    Examples
    --------
    Well-separated clusters are recovered at a workable budget.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = np.r_[rng.normal(-5, 0.4, (300, 2)), rng.normal(5, 0.4, (300, 2))]
    >>> r = dp_kmeans(X, k=2, epsilon=20.0, n_iter=5, bounds=(-8, 8), seed=1)
    >>> c = np.sort(r["centers"][:, 0])
    >>> bool(c[0] < -2 and c[1] > 2)
    True

    The budget divides across iterations, so each pass is cheaper the more
    passes are requested -- and the answer degrades accordingly.

    >>> a = dp_kmeans(X, k=2, epsilon=4.0, n_iter=2, bounds=(-8, 8), seed=1)
    >>> b = dp_kmeans(X, k=2, epsilon=4.0, n_iter=20, bounds=(-8, 8), seed=1)
    >>> bool(a["epsilon_per_iteration"] > b["epsilon_per_iteration"])
    True

    Labels cover every point.

    >>> int(r["labels"].size)
    600

    >>> dp_kmeans(X, k=0, epsilon=1.0)
    Traceback (most recent call last):
        ...
    ValueError: k must be at least 1
    """
    epsilon, _ = check_budget(epsilon)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, p = X.shape
    k = int(k)
    if k < 1:
        raise ValueError("k must be at least 1")
    n_iter = int(n_iter)
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1")
    if bounds is None:
        lo, hi = float(X.min()), float(X.max())
        warn = ["bounds were taken from the data, which is a non-private "
                "query; supply `bounds` from outside the data"]
    else:
        lo, hi = float(bounds[0]), float(bounds[1])
        warn = []
    Xc, lo, hi = clip_to_range(X, lo, hi)

    # Split across iterations AND between the sum and count queries.
    eps_iter = epsilon / (2.0 * n_iter)
    rng = np.random.default_rng(seed)
    centers = Xc[rng.choice(n, k, replace=False)].astype(float)
    reinit = 0
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        d2 = ((Xc[:, None] - centers[None]) ** 2).sum(-1)
        labels = np.argmin(d2, axis=1)
        for j in range(k):
            m = labels == j
            noisy_count = m.sum() + rng.laplace(0.0, 1.0 / eps_iter)
            noisy_sum = (Xc[m].sum(axis=0) if m.any() else np.zeros(p)) + \
                rng.laplace(0.0, (hi - lo) / eps_iter, p)
            if noisy_count < 1.0:
                centers[j] = Xc[rng.integers(n)]
                reinit += 1
            else:
                centers[j] = noisy_sum / noisy_count
    d2 = ((Xc[:, None] - centers[None]) ** 2).sum(-1)
    labels = np.argmin(d2, axis=1)
    return RichResult(
        title="DP k-means",
        summary_lines=[("epsilon", epsilon), ("k", k), ("iterations", n_iter),
                       ("epsilon/iteration", float(eps_iter))],
        warnings=warn + (["clusters were reinitialised from noisy counts below 1; "
                          "the budget is thin for this k"] if reinit else []),
        payload={
            "centers": centers, "labels": labels,
            "epsilon_per_iteration": float(eps_iter),
            "n_reinitialised": int(reinit),
            "inertia": float(np.sum(d2[np.arange(n), labels])),
            "epsilon": epsilon, "k": k, "n_iter": n_iter,
            "bounds": (lo, hi), "method": "dp_kmeans",
        },
    )


def cheatsheet():
    return "dpkmn: budget splits across ITERATIONS too, so more passes make it worse -- keep n_iter small"
