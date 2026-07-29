# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PCA preserves variance along the principal components."""

import numpy as np

from ._richresult import RichResult
from .hmpcac import geron_principal_components

__all__ = ["geron_pca_variance"]


def geron_pca_variance(X, n_components=None, threshold=0.95, n_probes=64, seed=0):
    """
    PCA preserves variance along principal components.

    Formula: max_w w^T Sigma w s.t. ||w||=1

    The components are DELEGATED to
    :func:`~morie.fn.hmpcac.geron_principal_components`; what this adds is
    the variance accounting Geron actually uses -- the cumulative ratio,
    and the smallest number of components reaching ``threshold`` of the
    total variance, which is how you choose d instead of guessing it.

    The maximisation claim is CHECKED, not asserted: ``probe_max`` is the
    largest w^T Sigma w found over ``n_probes`` unit vectors drawn from a
    reproducible integer LCG, and it never exceeds the first eigenvalue.

    Parameters
    ----------
    X : array-like, shape (m, p)
    n_components : int, optional
    threshold : float, default 0.95
        Variance fraction to reach.
    n_probes : int, default 64
        Random unit vectors used to check the maximisation.
    seed : int, default 0
        LCG seed for the probes.

    Returns
    -------
    result : RichResult
        Keys: explained_variance, explained_variance_ratio, cumulative,
        n_components_for_threshold, top_variance, probe_max, estimate, n,
        method.

    Examples
    --------
    A cloud spread only along x0 puts all its variance on one component;
    with values -2, 0, 2 that variance is 4:

    >>> X = [[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]]
    >>> r = geron_pca_variance(X)
    >>> [round(float(v), 12) for v in r["explained_variance_ratio"]]
    [1.0, 0.0]
    >>> round(float(r["top_variance"]), 12)
    4.0
    >>> int(r["n_components_for_threshold"])
    1

    No random direction beats the first component:

    >>> bool(r["probe_max"] <= r["top_variance"] + 1e-12)
    True

    References
    ----------
    Geron Ch 7
    """
    A = np.asarray(X, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"geron_pca_variance: X must be 2-D, got ndim={A.ndim}")
    thr = float(threshold)
    if not (0.0 < thr <= 1.0):
        raise ValueError(f"geron_pca_variance: threshold must lie in (0, 1], got {threshold!r}")
    npr = int(n_probes)
    if npr < 1:
        raise ValueError(f"geron_pca_variance: n_probes must be >= 1, got {n_probes!r}")

    base = geron_principal_components(A, n_components=n_components)
    var = np.asarray(base["explained_variance"], dtype=float)
    ratio = np.asarray(base["explained_variance_ratio"], dtype=float)
    cum = np.cumsum(ratio)
    reach = np.searchsorted(cum, thr - 1e-12) + 1
    reach = int(min(reach, ratio.size))

    Xc = A - A.mean(axis=0)
    Sigma = (Xc.T @ Xc) / (A.shape[0] - 1)
    p = A.shape[1]
    s = int(seed) % 2**32
    best = -np.inf
    for _ in range(npr):
        w = np.empty(p)
        for j in range(p):
            s = (1664525 * s + 1013904223) % 2**32
            w[j] = (s + 0.5) / 2**32 * 2.0 - 1.0
        nw = np.linalg.norm(w)
        if nw == 0:
            continue
        w /= nw
        best = max(best, float(w @ Sigma @ w))

    return RichResult(
        title="PCA variance",
        summary_lines=[
            ("Top component variance", float(var[0])),
            ("Cumulative at cut", float(cum[reach - 1])),
            ("Components for threshold", reach),
        ],
        interpretation="The first component is the unit direction of maximal variance; probes confirm it numerically.",
        payload={
            "explained_variance": var,
            "explained_variance_ratio": ratio,
            "cumulative": cum,
            "n_components_for_threshold": reach,
            "top_variance": float(var[0]),
            "probe_max": float(best),
            "covariance": Sigma,
            "estimate": ratio,
            "n": int(A.shape[0]),
            "method": "Variance accounting over principal components with a random-direction check",
        },
    )


def cheatsheet():
    return "hmpcav: PCA variance accounting along principal components"
