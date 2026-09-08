# morie.fn -- function file (rootcoder007/morie)
"""Multivariate kernel density estimate."""

from . import _array_core as np

from ._horowitz import kernel, silverman_bw
from ._richresult import RichResult

__all__ = ["hrz_kde_multivariate", "horowitz_multivariate_kde"]


def hrz_kde_multivariate(x, grid=None, h=None, kernel_name="gaussian"):
    r"""Product-kernel density estimate in d dimensions (Horowitz
    Ch. 2):

    .. math:: \hat f(x) = \frac{1}{n \prod_j h_j}\sum_i
              K\!\left(H^{-1}(x - X_i)\right),
              \qquad H = \mathrm{diag}(h).

    The rate degrades to :math:`n^{-2/(4+d)}`: the curse of
    dimensionality in its exact form. At d = 5 the rate is already
    :math:`n^{-2/9}`, which is why the book turns to index and
    additive restrictions rather than estimating high-dimensional
    densities. The effective rate is returned so the cost is explicit.

    Parameters
    ----------
    x : array-like, shape (n, d)
        Sample.
    grid : array-like, shape (m, d), optional
        Evaluation points; the sample itself if omitted.
    h : float or array-like, optional
        Per-dimension bandwidths.
    kernel_name : str
        Kernel.

    Returns
    -------
    RichResult
        keys: ``grid``, ``density``, ``bandwidths``,
        ``rate_exponent``, ``d``, ``n``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 2 (multivariate density estimation).
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] == 1 and X.shape[1] > 1:
        X = X.T
    n, d = X.shape
    if n < 2:
        raise ValueError("need at least 2 observations.")
    if h is None:
        hs = np.array([silverman_bw(X[:, j]) for j in range(d)])
    else:
        hs = np.atleast_1d(np.asarray(h, dtype=float))
        if hs.size == 1:
            hs = np.full(d, float(hs[0]))
        if hs.size != d:
            raise ValueError(f"h must have 1 or {d} entries.")
    if np.any(hs <= 0):
        raise ValueError("bandwidths must be positive.")
    G = X if grid is None else np.atleast_2d(np.asarray(grid, dtype=float))
    if G.shape[1] != d:
        raise ValueError(f"grid must have {d} columns.")
    dens = np.empty(G.shape[0])
    for i, pt in enumerate(G):
        u = (pt[None, :] - X) / hs[None, :]
        dens[i] = np.prod(kernel(u, kernel_name), axis=1).sum()
    dens /= n * np.prod(hs)
    return RichResult(payload={"grid": G, "density": dens, "bandwidths": hs,
                               "rate_exponent": -2.0 / (4.0 + d), "d": int(d),
                               "n": int(n),
                               "method": "Product kernel; rate n^{-2/(4+d)} -- the curse, exactly"})


def cheatsheet():
    return "hrzkd2: rate n^{-2/(4+d)}; d=5 already gives n^{-2/9}"


#: Catalogue alias for :func:`hrz_kde_multivariate`.
horowitz_multivariate_kde = hrz_kde_multivariate
