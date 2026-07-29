# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PCA projection via SVD-derived principal components."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_pca_projection"]

_METHOD = "PCA projection (SVD of the centred matrix)"


def geron_pca_projection(X, d):
    r"""Project onto the ``d`` leading principal components.

    .. math::
        Z = X_c W_d, \qquad
        X_c = U \Sigma V^{\top},\;
        W_d = \text{first } d \text{ columns of } V

    Centring first is not optional: without it the leading "component"
    just points at the mean, and the decomposition describes location
    instead of spread.  Working from the SVD of :math:`X_c` rather than
    an eigendecomposition of :math:`X_c^{\top}X_c` avoids squaring the
    condition number, which is what makes PCA numerically safe on
    near-collinear features.

    Component signs are arbitrary (:math:`v` and :math:`-v` span the same
    line), so they are fixed here by forcing the largest-magnitude entry
    of each component positive -- otherwise the same data gives
    sign-flipped scores on different machines.

    Parameters
    ----------
    X : array-like, shape (m, n)
    d : int
        Components to keep, ``1 <= d <= min(m, n)``.

    Returns
    -------
    RichResult
        Payload keys ``projection`` (m x d), ``components`` (d x n),
        ``explained_variance``, ``explained_variance_ratio``,
        ``cumulative_ratio``, ``mean``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 7, PCA section.

    Examples
    --------
    Points on the line ``y = x``: one component explains everything, and
    the scores are the signed distance along it (``sqrt(2)`` per unit).

    >>> X = [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]
    >>> r = geron_pca_projection(X, 1)
    >>> [round(v[0], 6) for v in r["projection"]]
    [-1.414214, 0.0, 1.414214]
    >>> round(r["explained_variance_ratio"][0], 10)
    1.0

    The component is the unit vector along that line:

    >>> [round(v, 6) for v in r["components"][0]]
    [0.707107, 0.707107]
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"X must be a non-empty (m, n) matrix, got shape {A.shape}.")
    if not np.all(np.isfinite(A)):
        raise ValueError("X contains non-finite values.")
    m, n = A.shape
    d = int(d)
    if not (1 <= d <= min(m, n)):
        raise ValueError(f"d must lie in [1, {min(m, n)}] (min(m, n)), got {d}.")

    mu = A.mean(axis=0)
    Xc = A - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:d].copy()
    for i in range(d):
        j = int(np.argmax(np.abs(V[i])))
        if V[i, j] < 0:
            V[i] = -V[i]
    Z = Xc @ V.T

    total = float(np.sum(S**2))
    var = (S[:d] ** 2) / max(m - 1, 1)
    ratio = (S[:d] ** 2) / total if total > 0 else np.zeros(d)

    return RichResult(
        title="PCA projection",
        summary_lines=[("Components", d), ("Cumulative variance", float(np.sum(ratio)))],
        payload={
            "projection": Z.tolist(),
            "components": V.tolist(),
            "explained_variance": var.tolist(),
            "explained_variance_ratio": ratio.tolist(),
            "cumulative_ratio": np.cumsum(ratio).tolist(),
            "singular_values": S[:d].tolist(),
            "mean": mu.tolist(),
            "estimate": Z.tolist(),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grpcap: centre, SVD, project on first d right singular vectors; signs pinned for determinism"
