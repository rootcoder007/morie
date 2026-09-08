# morie.fn -- wave 2 slice w2_00 (rootcoder007/morie)
"""Bias-corrected and accelerated (BCa) bootstrap confidence interval.

Source, read as rendered page images: Davison, A. C. and Hinkley, D. V.
(1997), *Bootstrap Methods and their Application*, Cambridge University
Press.  Page 204, equations (5.20)-(5.21):

    theta_alpha = t*_((R+1) alpha~),
    alpha~ = Phi( w + (w + z_alpha) / (1 - a (w + z_alpha)) ),

page 205, equation (5.22), the bias correction in simulation terms:

    w = Phi^-1( #{t*_r <= t} / (R + 1) ),

and page 209, equation (5.27), the nonparametric acceleration from the
empirical influence values l_j:

    a = (1/6) sum l_j^3 / ( sum l_j^2 )^{3/2}.

The empirical influence values are the jackknife ones,
l_j = (n - 1)(t - t_{-j}); for the sample mean that reduces to
l_j = y_j - ybar, exactly as the book states in Example 5.8.  That
example prints a = 0.0938 for the air-conditioning data
(3, 5, 7, 18, 43, 85, 91, 98, 100, 130, 230, 487), and that printed
number is the module's anchor.

Both conventions are returned.  ``lo``/``hi`` use type-7 quantiles at
alpha~, matching the rest of this shelf; ``lo_order``/``hi_order`` use
the book's ((R+1) alpha~)-th order statistic.  a = w = 0 collapses both
to the plain percentile interval, which is the second anchor.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["boot_bca_ci"]


def _order_stat(sv, r):
    """The r-th order statistic of the sorted replicates, r real, 1-based.

    Interpolated linearly and clamped at the ends, which is what the book
    means by "even with interpolation the relevant quantile cannot be
    calculated" when (R+1)alpha~ falls outside [1, R].
    """
    R = len(sv)
    if r <= 1.0:
        return sv[0]
    if r >= R:
        return sv[R - 1]
    lo = int(math.floor(r))
    fr = r - lo
    return sv[lo - 1] + fr * (sv[lo] - sv[lo - 1])


def boot_bca_ci(theta_hat, theta_b, x, stat, alpha=0.05):
    """BCa interval.

    Parameters
    ----------
    theta_hat : float
        The estimate on the original data.
    theta_b : array-like
        The R bootstrap replicates.
    x : array-like
        The original sample, used only for the jackknife.
    stat : callable
        The statistic; called on the leave-one-out samples.
    alpha : float
        Two-sided error rate.

    Returns
    -------
    lo, hi : the endpoints at type-7 quantiles of alpha~
    lo_order, hi_order : the book's order-statistic endpoints
    z0 : the bias correction w of equation (5.22)
    accel : the acceleration a of equation (5.27)
    """
    v = core.vec(theta_b)
    R = len(v)
    if R == 0:
        raise ValueError("boot_bca_ci: no bootstrap replicates")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("boot_bca_ci: alpha must lie strictly between 0 and 1")
    xx = core.vec(x)
    n = len(xx)
    if n < 2:
        raise ValueError("boot_bca_ci: need at least two observations for the jackknife")
    t = float(theta_hat)
    # (5.22): bias correction from the proportion of replicates at or below t
    cnt = 0
    for u in v:
        if u <= t:
            cnt += 1
    p = (cnt + 0.0) / (R + 1.0)
    if p <= 0.0:
        p = 0.5 / (R + 1.0)
    if p >= 1.0:
        p = 1.0 - 0.5 / (R + 1.0)
    w = core.qnorm(p)
    # (5.27) with jackknife empirical influence values
    s2 = 0.0
    s3 = 0.0
    for j in range(n):
        tj = float(stat([xx[i] for i in range(n) if i != j]))
        lj = (n - 1.0) * (t - tj)
        s2 += lj * lj
        s3 += lj * lj * lj
    acc = s3 / (6.0 * (s2 ** 1.5)) if s2 > 0.0 else 0.0
    out = {}
    sv = sorted(v)
    for nm, q in (("lo", a / 2.0), ("hi", 1.0 - a / 2.0)):
        z = core.qnorm(q)
        den = 1.0 - acc * (w + z)
        if den == 0.0:
            raise ValueError("boot_bca_ci: the acceleration makes the BCa transform singular")
        at = core.pnorm(w + (w + z) / den)
        out[nm + "_alpha"] = at
        out[nm] = core.quantile7(v, at)
        out[nm + "_order"] = _order_stat(sv, (R + 1.0) * at)
    return RichResult(
        title="BCa bootstrap interval",
        summary_lines=[("lo", out["lo"]), ("hi", out["hi"])],
        payload={
            "lo": out["lo"],
            "hi": out["hi"],
            "lo_order": out["lo_order"],
            "hi_order": out["hi_order"],
            "alpha_lo": out["lo_alpha"],
            "alpha_hi": out["hi_alpha"],
            "estimate": out["hi"] - out["lo"],
            "z0": w,
            "accel": acc,
            "B": R,
            "n": n,
            "method": "Davison and Hinkley (1997) eqs. (5.21), (5.22), (5.27)",
        },
    )


def cheatsheet():
    return "btbca: Bias-corrected accelerated (BCa) CI"


# compact alias per ledger/NAMING.md
bootbcaci = boot_bca_ci
