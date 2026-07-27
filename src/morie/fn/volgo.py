# morie.fn -- function file (rootcoder007/morie)
"""Orthogonal (principal-component) GARCH."""

import numpy as np

from ._garch import garch_fit
from ._richresult import RichResult

__all__ = ["vol_garch_orthogonal"]


def vol_garch_orthogonal(R_panel, k=None):
    r"""Orthogonal GARCH: univariate GARCH on principal components.

    Rotates the panel to its principal components, fits a univariate
    GARCH(1,1) to each, then rotates the diagonal component-variance
    matrix back:

    .. math:: H_t = W \, \mathrm{diag}(\sigma^2_{1t},\dots,
              \sigma^2_{kt}) \, W'.

    Keeping only k < d components makes the estimator tractable for
    wide panels, but the reconstructed :math:`H_t` is then rank k and
    singular -- so the returned ``full_rank`` flag says whether it can
    be inverted, rather than leaving a caller to discover it.

    Parameters
    ----------
    R_panel : array-like, shape (T, d)
        Return panel.
    k : int, optional
        Components retained; defaults to all d.

    Returns
    -------
    RichResult
        keys: ``H`` (T, d, d), ``component_sigma2`` (T, k),
        ``loadings`` (d, k), ``explained_variance_ratio``,
        ``full_rank``, ``k``, ``d``, ``T``, ``method``.

    References
    ----------
    Alexander, C. (2001). Orthogonal GARCH. In *Mastering Risk*,
    Vol. 2, Financial Times-Prentice Hall, 21-38.

    Tsay, R. S. (2010). *Analysis of Financial Time Series* (3rd ed.).
    Wiley. Ch. 10.
    """
    R = np.asarray(R_panel, dtype=float)
    if R.ndim != 2:
        raise ValueError("R_panel must be 2-D (T observations x d series).")
    T, d = R.shape
    if d < 2:
        raise ValueError("need at least 2 series.")
    if T < 50:
        raise ValueError(f"need at least 50 observations, got {T}.")
    if not np.all(np.isfinite(R)):
        raise ValueError("R_panel must be finite.")
    k = d if k is None else int(k)
    if not 1 <= k <= d:
        raise ValueError(f"k must lie in 1..{d}, got {k}.")

    E = R - R.mean(axis=0)
    S = np.cov(E, rowvar=False)
    w, V = np.linalg.eigh(S)
    order = np.argsort(w)[::-1][:k]
    lam = w[order]
    W = V[:, order]
    pcs = E @ W

    s2 = np.column_stack([garch_fit(pcs[:, j], "garch")["sigma2"] for j in range(k)])
    H = np.einsum("ij,tj,kj->tik", W, s2, W)

    return RichResult(
        payload={
            "H": H, "component_sigma2": s2, "loadings": W,
            "explained_variance_ratio": lam / w.sum(),
            "full_rank": bool(k == d), "k": int(k), "d": int(d), "T": int(T),
            "method": "Orthogonal GARCH: univariate GARCH(1,1) on principal components",
        }
    )


def cheatsheet():
    return "volgo: PCA rotate, GARCH each PC, rotate back; rank k if k < d"
