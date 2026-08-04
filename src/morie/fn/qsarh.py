# morie.fn -- tail3 batch (rootcoder007/morie)
"""Hansch QSAR (rho-sigma-pi analysis).

Source consulted: Hansch, C. & Fujita, T. (1964). rho-sigma-pi analysis.  A
method for the correlation of biological activity and chemical structure.
*Journal of the American Chemical Society* 86(8), 1616-1626, doi
10.1021/ja01062a035.  Biological potency, expressed as log(1/C) for the
concentration C producing a standard response, is regressed on a hydrophobic
term, Hammett's electronic constant sigma and optionally a steric term.  The
extended-range (parabolic) form of the Hansch equation is

    log(1/C) = -a (log P)^2 + b log P + rho sigma + k

whose maximum is at log P_0 = b / (2 a), the optimum partition coefficient.
Setting ``parabolic=False`` fits the linear free-energy form without the
quadratic term.  The substituent constant pi may be passed in place of log P.
"""

from __future__ import annotations

from . import _array_core as np

from . import t3util as _t3
from ._richresult import RichResult

__all__ = ["hansch_qsar"]


def hansch_qsar(activities, logP, sigma=None, es=None, parabolic=True):
    """Fit the Hansch equation by least squares.

    Parameters
    ----------
    activities : array-like
        log(1/C) values.
    logP : array-like
        Partition coefficient log P (or the substituent constant pi).
    sigma : array-like, optional
        Hammett electronic substituent constant.
    es : array-like, optional
        Taft steric substituent constant.
    parabolic : bool
        Include the -(log P)^2 term.

    Returns
    -------
    RichResult
        estimate (fitted intercept k), coefficients, r2, s, logp0, rho, n,
        method.

    References
    ----------
    Hansch & Fujita (1964), J. Am. Chem. Soc. 86(8), 1616-1626.
    """
    y = np.atleast_1d(np.asarray(activities, dtype=float)).ravel()
    lp = np.atleast_1d(np.asarray(logP, dtype=float)).ravel()
    n = int(min(y.size, lp.size))
    cols = [[1.0] * n, [float(lp[i]) for i in range(n)]]
    names = ["k", "logP"]
    if parabolic:
        cols.append([float(lp[i]) ** 2 for i in range(n)])
        names.append("logP2")
    if sigma is not None:
        sg = np.atleast_1d(np.asarray(sigma, dtype=float)).ravel()
        cols.append([float(sg[i]) for i in range(n)])
        names.append("sigma")
    if es is not None:
        et = np.atleast_1d(np.asarray(es, dtype=float)).ravel()
        cols.append([float(et[i]) for i in range(n)])
        names.append("Es")
    X = np.asarray([[cols[c][i] for c in range(len(cols))] for i in range(n)], dtype=float)
    yv = np.asarray([float(y[i]) for i in range(n)], dtype=float)
    beta = _t3.ols(X, yv)
    fit = X @ beta
    resid = yv - fit
    rss = float(np.sum(resid * resid))
    tss = float(np.sum((yv - float(np.mean(yv))) ** 2))
    r2 = 1.0 - rss / tss if tss > 0.0 else float("nan")
    p = int(len(cols))
    s = float(np.sqrt(rss / (n - p))) if n > p else float("nan")
    b = float(beta[1])
    a = -float(beta[2]) if parabolic else 0.0
    logp0 = b / (2.0 * a) if a != 0.0 else float("nan")
    rho = float(beta[names.index("sigma")]) if "sigma" in names else float("nan")
    return RichResult(
        payload={
            "estimate": float(beta[0]),
            "coefficients": beta,
            "names": names,
            "r2": float(r2),
            "s": s,
            "rss": rss,
            "logp0": logp0,
            "rho": rho,
            "n": n,
            "method": "Hansch rho-sigma-pi QSAR (Hansch & Fujita 1964)",
        }
    )


# CANONICAL TEST
# >>> # exactly parabolic data is fitted with r2 = 1 and logP0 recovered
# >>> lp = [0.0, 1.0, 2.0, 3.0, 4.0]
# >>> y = [-(v - 2.0) ** 2 + 5.0 for v in lp]
# >>> r = hansch_qsar(y, lp)
# >>> assert abs(r["r2"] - 1.0) < 1e-9
# >>> assert abs(r["logp0"] - 2.0) < 1e-9


def cheatsheet():
    return "qsarh(activities, logP, sigma): Hansch parabolic QSAR fit."


# compact alias per ledger/NAMING.md (registered in _lazy_map.json)
hanschqsar = hansch_qsar
