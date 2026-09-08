# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Silhouette score: mean over points of (b - a) / max(a, b)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_silhouette_score"]

_METHOD = "Silhouette score"


def geron_silhouette_score(X, labels):
    r"""How well each point sits in its cluster relative to the next-best one.

    .. math::
        s_i = \frac{b_i - a_i}{\max(a_i, b_i)}, \qquad
        \text{silhouette} = \frac{1}{m}\sum_i s_i

    :math:`a_i` is the mean distance to the *other* points of the same
    cluster -- excluding the point itself, which is why a singleton
    cluster has no defined :math:`a` and is scored 0 by convention rather
    than by division by zero.  :math:`b_i` is the mean distance to the
    nearest other cluster.  The score lives in ``[-1, 1]`` and needs no
    ground truth, which is what makes it usable for choosing ``k``:
    unlike inertia it does not fall monotonically as clusters multiply.

    Parameters
    ----------
    X : array-like, shape (m, n)
    labels : array-like of int, shape (m,)
        At least two distinct clusters are required.

    Returns
    -------
    RichResult
        Payload keys ``silhouette``, ``per_sample``, ``per_cluster``,
        ``a``, ``b``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 8, Silhouette Score section.

    Examples
    --------
    Two tight, far-apart pairs on a line.  For the outer points
    ``a = 1`` and ``b = 10.5`` (score ``9.5/10.5``); for the inner ones
    ``b = 9.5`` (score ``8.5/9.5``), and the mean of the four is
    0.899749.

    >>> X = [[0.0], [1.0], [10.0], [11.0]]
    >>> r = geron_silhouette_score(X, [0, 0, 1, 1])
    >>> round(r["silhouette"], 6)
    0.899749
    >>> [round(v, 6) for v in r["a"]]
    [1.0, 1.0, 1.0, 1.0]

    Shuffling the labels so clusters interleave destroys the score:

    >>> round(geron_silhouette_score(X, [0, 1, 0, 1])["silhouette"], 6)
    -0.45
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    lab = np.asarray(labels).ravel()
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"X must be a non-empty (m, n) matrix, got shape {A.shape}.")
    if lab.size != A.shape[0]:
        raise ValueError(f"labels has {lab.size} entries but X has {A.shape[0]} rows.")
    if not np.all(np.isfinite(A)):
        raise ValueError("X contains non-finite values.")
    uniq = np.unique(lab)
    if uniq.size < 2:
        raise ValueError(
            f"silhouette needs at least 2 clusters, got {uniq.size}."
        )
    if uniq.size >= A.shape[0]:
        raise ValueError(
            f"{uniq.size} clusters for {A.shape[0]} points leaves no within-cluster distances."
        )

    D = np.sqrt(((A[:, None, :] - A[None, :, :]) ** 2).sum(axis=2))
    m = A.shape[0]
    a = np.zeros(m)
    b = np.zeros(m)
    s = np.zeros(m)
    for i in range(m):
        own = lab == lab[i]
        n_own = int(own.sum())
        if n_own <= 1:
            a[i] = 0.0
            b[i] = min(D[i, lab == c].mean() for c in uniq if c != lab[i])
            s[i] = 0.0
            continue
        a[i] = (D[i, own].sum()) / (n_own - 1)
        b[i] = min(D[i, lab == c].mean() for c in uniq if c != lab[i])
        denom = max(a[i], b[i])
        s[i] = 0.0 if denom == 0 else (b[i] - a[i]) / denom

    per_cluster = {int(c): float(s[lab == c].mean()) for c in uniq}
    return RichResult(
        title="Silhouette score",
        summary_lines=[("Silhouette", float(s.mean())), ("Clusters", int(uniq.size))],
        payload={
            "silhouette": float(s.mean()),
            "per_sample": s.tolist(),
            "per_cluster": per_cluster,
            "a": a.tolist(),
            "b": b.tolist(),
            "estimate": float(s.mean()),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grsil: s_i = (b-a)/max(a,b), a excludes the point itself; singleton clusters score 0"
