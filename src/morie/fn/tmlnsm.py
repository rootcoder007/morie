# morie.fn -- function file (rootcoder007/morie)
"""TMLE for a non-smooth functional: the counterfactual median difference."""

import math

from . import _s04core as S
from . import _tail1core as C
from .tmlmpi import _cdf_bank

from ._richresult import RichResult

__all__ = ["tmle_non_smooth"]


def tmle_non_smooth(y, D, X, bw):
    """Targeted difference of counterfactual medians.

    A median is not a smooth functional of the distribution: it has no
    influence curve until the density at the median exists, and the
    plug-in through an untargeted CDF inherits first-order bias.  The
    fix used here is the standard one -- target the CDF, then invert.
    ``F_a(t)`` is targeted at every distinct observed outcome value, the
    median is read off by linear interpolation of the targeted CDF at
    1/2, and the influence curve is the smoothed one

        ``IC_{m_a}(O) = -IC_{F_a(m_a)}(O) / f_a(m_a)``,

    with ``f_a`` a Gaussian kernel density built from the targeted
    counterfactual increments at bandwidth ``bw``.  The bandwidth is a
    real knob, not a formality: it sets how much the reported SE trusts
    the local slope of the CDF, and a bandwidth much smaller than the
    grid spacing will make ``f_a`` collapse and the SE explode.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like, shape (n,)
        Binary treatment.
    X : array-like, shape (n, p)
        Covariates.
    bw : float
        Kernel bandwidth for the density at the median; must be positive.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``m1``, ``m0``, ``f1``, ``f0``, ``n``.

    References
    ----------
    Diaz, I. (2017).  Efficient estimation of quantiles in missing data
    models.  Journal of Statistical Planning and Inference 190:39-51.
    doi:10.1016/j.jspi.2017.05.001.  The targeting step is van der Laan,
    M. J. & Rubin, D. (2006), IJB 2(1):11.
    """
    yv = C.vec(y)
    Dv = C.vec(D)
    n = len(yv)
    bw = float(bw)
    if n == 0 or len(Dv) != n:
        raise ValueError("tmle_non_smooth: y and D must share one length")
    if not bw > 0.0:
        raise ValueError("tmle_non_smooth: bw must be positive")
    Xm = C.mat(X)
    if len(Xm) != n:
        raise ValueError("tmle_non_smooth: X must have one row per subject")
    W = [[1.0] + list(Xm[i]) for i in range(n)]
    gb = S.glmbin(W, Dv)
    g = [S.clip(S.expit(C.dot(W[i], gb)), 0.025, 0.975) for i in range(n)]
    grid = sorted(set(yv))
    K = len(grid)
    F, IC = _cdf_bank(yv, Dv, W, g, grid)

    def invert(a):
        j = K - 1
        for k in range(K):
            if F[a][k] >= 0.5:
                j = k
                break
        if j == 0:
            return grid[0], 0, 0.0
        f0 = F[a][j - 1]
        f1 = F[a][j]
        w = 0.0 if f1 <= f0 else (0.5 - f0) / (f1 - f0)
        return grid[j - 1] + w * (grid[j] - grid[j - 1]), j, w

    out = []
    for a in (0, 1):
        m, j, w = invert(a)
        dens = 0.0
        for k in range(K):
            inc = F[a][k] - (F[a][k - 1] if k > 0 else 0.0)
            u = (m - grid[k]) / bw
            dens += inc * math.exp(-0.5 * u * u) / (bw * math.sqrt(2.0 * math.pi))
        if dens < 1e-12:
            raise ValueError("tmle_non_smooth: kernel density at the median is zero; widen bw")
        icf = [(1.0 - w) * IC[a][j - 1][i] + w * IC[a][j][i] if j > 0 else IC[a][0][i]
               for i in range(n)]
        out.append((m, dens, [-v / dens for v in icf]))
    est = out[1][0] - out[0][0]
    ic = [out[1][2][i] - out[0][2][i] for i in range(n)]
    mn = sum(ic) / n
    se = math.sqrt(sum((v - mn) ** 2 for v in ic) / (n - 1) / n) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": est, "se": se, "m1": out[1][0], "m0": out[0][0],
        "f1": out[1][1], "f0": out[0][1], "n": n,
        "method": "TMLE for the counterfactual median difference"})


def cheatsheet():
    return "tmlnsm: TMLE for a non-smooth functional (median difference)."

# public names resolved by fn/_lazy_map.json
tmlenonsmooth = tmle_non_smooth
