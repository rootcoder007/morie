# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Explained variance ratio per principal component."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_explained_variance_ratio"]


def geron_explained_variance_ratio(X, n_components=None, center=True):
    """
    Explained variance ratio per principal component.

    Formula: EVR_k = sigma_k^2 / sum_j sigma_j^2

    The singular values come from an SVD of the *centred* matrix, which is
    what makes them PCA singular values rather than plain ones; skipping
    the centring silently changes the first component into the data mean
    direction, so ``center`` is exposed and defaults to True.

    The ratios sum to 1 across all components, so ``cumulative`` reads
    directly as "fraction of variance kept by the first k components", and
    ``n_components_for(0.95)`` is answered by ``n_for_95``.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Data matrix, one row per instance.
    n_components : int, optional
        Truncate the report to the leading ``k`` components.
    center : bool, default True
        Subtract the column means first.

    Returns
    -------
    result : RichResult
        Keys: explained_variance_ratio, explained_variance,
        singular_values, cumulative, n_for_95, components, estimate,
        n, method.

    Examples
    --------
    Data lying exactly on a line puts all the variance in one component:

    >>> X = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    >>> r = geron_explained_variance_ratio(X)
    >>> [round(v, 12) for v in r["explained_variance_ratio"]]
    [1.0, 0.0]
    >>> r["n_for_95"]
    1

    Axis-aligned data with variances 4 and 1 splits 80/20:

    >>> Y = [[-2.0, -1.0], [2.0, 1.0], [-2.0, 1.0], [2.0, -1.0]]
    >>> r2 = geron_explained_variance_ratio(Y)
    >>> [round(v, 12) for v in r2["explained_variance_ratio"]]
    [0.8, 0.2]
    >>> [round(v, 12) for v in r2["explained_variance"]]
    [5.333333333333, 1.333333333333]
    >>> [round(v, 12) for v in r2["cumulative"]]
    [0.8, 1.0]

    References
    ----------
    Géron Ch 7
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_explained_variance_ratio: X must be a non-empty 2-D array, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_explained_variance_ratio: X contains non-finite values")
    m, n = A.shape
    if m < 2:
        raise ValueError(f"geron_explained_variance_ratio: need at least 2 instances to have variance, got {m}")

    Ac = A - A.mean(axis=0) if center else A
    U, s, Vt = np.linalg.svd(Ac, full_matrices=False)
    var = s**2 / (m - 1)
    total = float(var.sum())
    if total <= 0:
        raise ValueError("geron_explained_variance_ratio: X has zero total variance; the ratio is undefined")
    evr = var / total
    cum = np.cumsum(evr)

    k = len(evr) if n_components is None else int(n_components)
    if k < 1 or k > len(evr):
        raise ValueError(f"geron_explained_variance_ratio: n_components must lie in 1..{len(evr)}, got {n_components!r}")

    return RichResult(
        title="Explained variance ratio",
        summary_lines=[("Components", int(k)), ("Variance kept", float(cum[k - 1]))],
        tables=[{
            "title": "components",
            "headers": ["k", "singular value", "variance", "ratio", "cumulative"],
            "rows": [[i + 1, float(s[i]), float(var[i]), float(evr[i]), float(cum[i])] for i in range(k)],
        }],
        interpretation="Ratios sum to 1 over all components, so `cumulative` is the fraction of variance kept.",
        payload={
            "explained_variance_ratio": evr[:k].tolist(),
            "explained_variance": var[:k].tolist(),
            "singular_values": s[:k].tolist(),
            "cumulative": cum[:k].tolist(),
            "total_variance": total,
            "n_for_95": int(np.searchsorted(cum, 0.95) + 1),
            "components": Vt[:k].tolist(),
            "centered": bool(center),
            "estimate": float(evr[0]),
            "n": int(m),
            "method": "EVR from the SVD of the centred data matrix",
        },
    )


def cheatsheet():
    return "hmevr: Explained variance ratio per principal component"
