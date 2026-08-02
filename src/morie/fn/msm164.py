# morie.fn -- function file (rootcoder007/morie)
"""Hyperplane equation and point classification (MVSML Eq. 9.1)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["hyperplane_side"]


def hyperplane_side(X, beta, beta0=0.0):
    r"""Evaluate a separating hyperplane at one or more points.

    .. math:: f(x) = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p

    Points with :math:`f(x) = 0` lie ON the (p-1)-dimensional
    hyperplane (Eq. 9.1 of the source, extended to p dimensions by its
    Eq. 9.2); the sign of :math:`f(x)` says which side, and
    :math:`|f(x)| / \lVert\beta\rVert` is the Euclidean distance to
    the plane -- the margin quantity of an SVM. This replaces a
    placeholder whose signature was six words of extracted prose and
    whose body returned the mean of the first argument. The generated
    name promised "ridge lasso elastic", but the chapter's Eq. (9.1)
    is the hyperplane definition; the module now implements what its
    source actually says.

    Parameters
    ----------
    X : array-like, shape (n, p) or (p,)
        Point(s) to evaluate.
    beta : array-like, shape (p,)
        Hyperplane normal vector; must not be all zero.
    beta0 : float, default 0.0
        Intercept.

    Returns
    -------
    RichResult
        keys: ``value`` (f(x) per point), ``side`` (sign),
        ``distance`` (|f| / ||beta||), ``on_plane``, ``n``,
        ``method``.

    References
    ----------
    Multivariate Statistical Machine Learning Methods for Genomic
    Prediction (2022). Springer. Ch. 9, Eqs. (9.1)-(9.2) (hyperplane
    definition), citing James et al. (2013), *An Introduction to
    Statistical Learning*, Ch. 9.
    """
    b = np.asarray(beta, dtype=float).ravel()
    nb = float(np.linalg.norm(b))
    if nb == 0:
        raise ValueError("beta must not be the zero vector; it defines no hyperplane.")
    P = np.asarray(X, dtype=float)
    single = P.ndim == 1
    if single:
        P = P.reshape(1, -1)
    if P.shape[1] != b.size:
        raise ValueError(f"X has {P.shape[1]} coordinates but beta has {b.size}.")
    f = float(beta0) + P @ b
    dist = np.abs(f) / nb
    side = np.sign(f)
    out = {
        "value": float(f[0]) if single else f,
        "side": float(side[0]) if single else side,
        "distance": float(dist[0]) if single else dist,
        "on_plane": bool(np.isclose(f[0], 0.0)) if single else np.isclose(f, 0.0),
        "n": int(P.shape[0]),
        "method": "Hyperplane f(x) = beta0 + beta'x (MVSML Eq. 9.1)",
    }
    return RichResult(payload=out)


# Back-compatible alias under the generated export name.
mvsml_ridge_lasso_elastic_eq_9_1 = hyperplane_side


def cheatsheet():
    return "msm164: hyperplane evaluation f(x) = beta0 + beta'x (MVSML Eq. 9.1)"
