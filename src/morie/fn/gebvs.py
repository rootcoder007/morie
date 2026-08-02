# morie.fn -- function file (rootcoder007/morie)
"""Genomic estimated breeding values."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["genomic_ebv", "gebv_selection"]


def genomic_ebv(marker_matrix, y=None, effects=None, h2=None,
                n_select=None, method="vanraden1"):
    r"""GEBVs and the selection decision they support.

    With the genomic relationship matrix of VanRaden method 1,

    .. math::
       G = \frac{ZZ'}{2\sum_j p_j(1-p_j)},
       \qquad Z_{ij} = M_{ij} - 2p_j,

    the GEBV is :math:`\hat g = G(G + \lambda I)^{-1} y` with
    :math:`\lambda = (1-h^2)/h^2`, which is ridge regression on the
    relationship scale (GBLUP).

    The DENOMINATOR is what makes :math:`G` comparable to a pedigree
    relationship matrix rather than an arbitrary kernel, and centring
    :math:`M` by :math:`2p_j` is what makes the diagonal average one.
    Skip the centring and :math:`G` no longer estimates relatedness at
    all.

    Selection is where the estimate meets its purpose, and two
    quantities decide whether it is worth acting on. ACCURACY,
    :math:`\mathrm{corr}(\hat g, g)`, sets the response to selection
    through the breeder's equation :math:`R = i\,r\,\sigma_g`; a GEBV
    with accuracy 0.3 delivers under a third of the gain of one with
    accuracy 0.95, whatever its rank ordering looks like. And selecting
    the top :math:`n` on :math:`\hat g` alone maximises immediate gain
    while shrinking the effective population size, so
    ``mean_relatedness_selected`` reports the relatedness among the
    chosen -- the inbreeding that a purely truncation-based selection
    accumulates.

    Parameters
    ----------
    marker_matrix : array-like, shape (n, m)
        Genotypes coded 0/1/2.
    y : array-like, shape (n,), optional
        Phenotypes; enables GBLUP.
    effects : array-like, shape (m,), optional
        Known marker effects; GEBV is then :math:`Z\beta` directly.
    h2 : float, optional
        Narrow-sense heritability, setting the shrinkage.
    n_select : int, optional
    method : {'vanraden1'}

    Returns
    -------
    RichResult
        ``gebv``, ``ranking``, ``selected``, ``G``,
        ``mean_relatedness_selected``, ``accuracy``,
        ``selection_intensity``, ``expected_response``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), sections 2.4
    and 2.5, pp. 49-57.
    VanRaden (2008), *Journal of Dairy Science* 91:4414-4423.
    Falconer and Mackay (1996) for the breeder's equation.

    Examples
    --------
    >>> import numpy as np
    >>> M = np.array([[0, 1, 2], [2, 1, 0], [1, 1, 1], [0, 0, 2]], float)
    >>> out = genomic_ebv(M, effects=[1.0, 0.0, -1.0])
    >>> int(out["ranking"][0])
    1
    """
    M = np.atleast_2d(np.asarray(marker_matrix, dtype=float))
    n, m = M.shape
    if method != "vanraden1":
        raise ValueError("method must be 'vanraden1', got %r." % method)
    if np.any(M < 0) or np.any(M > 2):
        raise ValueError("markers must be coded 0, 1 or 2.")
    p = M.mean(axis=0) / 2.0                    # allele frequencies
    Z = M - 2.0 * p[None, :]                    # centring is not optional
    denom = 2.0 * float(np.sum(p * (1.0 - p)))
    if denom <= 0:
        raise ValueError(
            "every marker is monomorphic; the relationship matrix is "
            "undefined."
        )
    G = Z @ Z.T / denom

    acc = np.nan
    if effects is not None:
        b = np.asarray(effects, dtype=float).ravel()
        if b.size != m:
            raise ValueError("effects has %d entries for %d markers."
                             % (b.size, m))
        g = Z @ b
    elif y is not None:
        yv = np.asarray(y, dtype=float).ravel()
        if yv.size != n:
            raise ValueError("y has %d entries for %d individuals."
                             % (yv.size, n))
        h = 0.5 if h2 is None else float(h2)
        if not 0 < h < 1:
            raise ValueError("h2 must lie in (0, 1), got %r." % h2)
        lam = (1 - h) / h
        A = G + lam * np.eye(n)
        yc = yv - yv.mean()
        g = G @ np.linalg.solve(A, yc)
        # accuracy from the prediction error variance of the mixed model
        Ci = np.linalg.inv(A)
        pev = np.diag(G) - np.diag(G @ Ci @ G)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.sqrt(np.clip(1.0 - pev / np.maximum(np.diag(G), 1e-12),
                                0.0, 1.0))
        acc = float(np.mean(r))
    else:
        raise ValueError("supply y (for GBLUP) or effects.")

    order = np.argsort(g)[::-1]
    k = int(n_select) if n_select else max(int(round(0.1 * n)), 1)
    k = min(max(k, 1), n)
    sel = order[:k]
    frac = k / n
    # selection intensity for truncation selection at proportion frac
    zq = _z(1 - frac)
    i = float(np.exp(-0.5 * zq ** 2) / np.sqrt(2 * np.pi) / frac) \
        if 0 < frac < 1 else 0.0
    sg = float(np.std(g, ddof=1)) if n > 1 else 0.0
    sub = G[np.ix_(sel, sel)]
    off = sub[~np.eye(k, dtype=bool)] if k > 1 else np.array([np.nan])
    return RichResult(
        payload={
            "estimate": g,
            "gebv": g,
            "ranking": order,
            "selected": sel,
            "G": G,
            "G_diagonal_mean": float(np.mean(np.diag(G))),
            "G_note": (
                "markers are centred by 2p before the cross-product and the "
                "sum is scaled by 2 sum p(1-p); without both, G is an "
                "arbitrary kernel rather than an estimate of relatedness, "
                "and its diagonal no longer averages one"
            ),
            "mean_relatedness_selected": float(np.mean(off)),
            "relatedness_note": (
                "selecting the top n on GEBV alone maximises immediate gain "
                "and shrinks the effective population; this is the inbreeding "
                "that truncation selection accumulates"
            ),
            "accuracy": acc,
            "selection_intensity": i,
            "selected_fraction": float(frac),
            "expected_response": (float(i * acc * sg)
                                  if acc == acc else float(i * sg)),
            "response_note": (
                "breeder's equation R = i r sigma_g; accuracy enters "
                "linearly, so a GEBV with r = 0.3 delivers under a third of "
                "the gain of one with r = 0.95 however its ranking looks"
            ),
            "n_markers": int(m),
            "n": int(n),
            "method": "Genomic estimated breeding value (VanRaden method 1)",
        }
    )


def _z(q):
    import math
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(-mid / math.sqrt(2.0)) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cheatsheet():
    return (
        "gebvs: GBLUP breeding values on VanRaden's G, with accuracy, "
        "selection intensity and the relatedness selection accumulates"
    )


#: Catalogue alias for :func:`genomic_ebv`.
gebv_selection = genomic_ebv
