# morie.fn -- function file (rootcoder007/morie)
"""Autoencoder reconstruction-error anomaly score."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["autoencoder_anomaly"]


def autoencoder_anomaly(X, k=2, n_iter=300, lr=0.05, seed=0, contamination=0.05):
    r"""Score anomalies by how badly a low-rank autoencoder reconstructs them.

    Fits a linear bottleneck :math:`X \approx XWW^\top` with :math:`W` of rank
    ``k`` and scores each row by :math:`\lVert x - \hat x\rVert^2`. Points
    lying off the dominant subspace reconstruct poorly and score high.

    The linear case is exactly PCA reconstruction error, which is worth
    stating because it bounds what an autoencoder buys: nonlinearity, not
    magic. A deep autoencoder finds curved manifolds a linear one cannot, but
    both share the fatal property below.

    **The anomalies are in the training data.** Unlike a supervised detector,
    the autoencoder is fitted on the contaminated sample, so it learns to
    reconstruct the anomalies too. With enough capacity it reconstructs
    everything perfectly and the score becomes uninformative -- more capacity
    makes detection *worse*, which inverts the usual intuition. The bottleneck
    is the regulariser doing the work, and ``k`` must stay well below the
    intrinsic dimension.

    Parameters
    ----------
    X : array-like
        Data ``(n, d)``.
    k : int
        Bottleneck rank, well below ``d``.
    n_iter, lr : int, float
        Gradient-descent controls.
    seed : int
        Seed.
    contamination : float
        Assumed anomaly fraction, used only for the flag threshold.

    Returns
    -------
    RichResult
        ``score``, ``rank``, ``anomaly``, ``reconstruction``,
        ``explained_fraction``.

    References
    ----------
    Sakurada, M., & Yairi, T. (2014). Anomaly detection using autoencoders
        with nonlinear dimensionality reduction. *MLSDA 2014*, 4-11.

    Examples
    --------
    A point off the dominant subspace scores highest.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> Z = rng.normal(size=(300, 1))
    >>> X = np.c_[Z, 2 * Z, -Z] + rng.normal(0, 0.05, (300, 3))
    >>> X = np.r_[X, [[3.0, -3.0, 3.0]]]
    >>> r = autoencoder_anomaly(X, k=1)
    >>> int(np.argmax(r["score"]))
    300

    More bottleneck capacity makes detection worse, not better -- the model
    learns to reconstruct the anomaly too.

    >>> tight = autoencoder_anomaly(X, k=1)["score"]
    >>> loose = autoencoder_anomaly(X, k=3)["score"]
    >>> bool(tight[300] / np.median(tight[:300]) > loose[300] / np.median(loose[:300]))
    True

    >>> autoencoder_anomaly(X, k=0)
    Traceback (most recent call last):
        ...
    ValueError: k must be between 1 and 3
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, d = X.shape
    k = int(k)
    if not 1 <= k <= d:
        raise ValueError(f"k must be between 1 and {d}")
    mu = X.mean(axis=0)
    Z = X - mu
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, (d, k))
    for _ in range(int(n_iter)):
        rec = Z @ W @ W.T
        E = rec - Z
        G = 2.0 * (E.T @ (Z @ W) + Z.T @ (E @ W)) / n
        W -= lr * G
        W, _ = np.linalg.qr(W)
    rec = Z @ W @ W.T
    err = ((Z - rec) ** 2).sum(axis=1)
    order = np.argsort(-err, kind="stable")
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    cut = float(np.quantile(err, 1.0 - contamination))
    total = float((Z**2).sum())
    return RichResult(
        title="Autoencoder anomaly score",
        summary_lines=[("n", n), ("d", d), ("k", k),
                       ("explained", 1.0 - float(err.sum()) / max(total, 1e-300))],
        warnings=["the anomalies are in the training data, so more bottleneck "
                  "capacity makes detection WORSE; keep k well below the "
                  "intrinsic dimension"],
        payload={
            "score": err, "rank": rank, "anomaly": err > cut,
            "reconstruction": rec + mu, "threshold": cut,
            "explained_fraction": 1.0 - float(err.sum()) / max(total, 1e-300),
            "W": W, "k": k, "method": "autoencoder_anomaly",
        },
    )


def cheatsheet():
    return "aE_an: linear case IS PCA error; anomalies are in the training data so MORE capacity is worse"
