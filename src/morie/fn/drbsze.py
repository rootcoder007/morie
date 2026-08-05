# morie.fn -- function file (rootcoder007/morie)
"""Finite-sample size correction for the doubly robust DiD estimator.

The point estimate and the influence function psi_i come from the
doubly robust moment of Sant'Anna, P. H. C. and Zhao, J. (2020),
*Journal of Econometrics* 219(1), 101-122, eq. (2.6).  Its asymptotic
standard error sqrt(sum psi^2)/n and the normal critical value give a
test whose actual size exceeds its nominal size when n is small or the
treated group is thin, because the nuisance parameters cost degrees of
freedom that the asymptotic formula does not charge for.

Two corrections are applied, both standard and both stated here so the
provenance is not overclaimed:

  1. a degrees-of-freedom inflation sqrt(n / (n - k)), k the number of
     columns of the outcome design, which is the classical HC1 scaling
     of MacKinnon, J. G. and White, H. (1985), *Journal of
     Econometrics* 29(3), 305-325;
  2. a Student-t critical value on min(n_treated, n_control) - 1
     degrees of freedom in place of the normal quantile, the
     small-cluster convention of Bell, R. M. and McCaffrey, D. F.
     (2002), *Survey Methodology* 28(2), 169-181.

Both are strictly conservative: the corrected interval contains the
uncorrected one, and as n grows the inflation tends to one and the t
quantile to the normal quantile, which is the limiting check.

NOTE ON PROVENANCE: the stub named "Roth-Sant'Anna (2023)".  That paper
(Econometrica 91(2), 737-747, doi:10.3982/ECTA19402) is about parallel
trends under a functional-form change and does not supply a
finite-sample size correction, so it is not cited as the source of the
formulas above.
"""

from __future__ import annotations

import math

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["dr_did_size_correction"]


def _tquant(p, df):
    """Two-sided Student-t quantile by Cornish-Fisher on the normal."""
    z = k.qnorm(p)
    g1 = (z ** 3 + z) / 4.0
    g2 = (5.0 * z ** 5 + 16.0 * z ** 3 + 3.0 * z) / 96.0
    g3 = (3.0 * z ** 7 + 19.0 * z ** 5 + 17.0 * z ** 3 - 15.0 * z) / 384.0
    g4 = (79.0 * z ** 9 + 776.0 * z ** 7 + 1482.0 * z ** 5
          - 1920.0 * z ** 3 - 945.0 * z) / 92160.0
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3 + g4 / df ** 4


def dr_did_size_correction(y, D, X=None, alpha=0.05):
    """DR-DiD with a degrees-of-freedom inflated SE and a t critical value.

    Parameters
    ----------
    y : array-like
        Outcome change dY = Y_post - Y_pre, one entry per unit.
    D : array-like
        Binary treatment indicator.
    X : 2-D array-like, optional
        Baseline covariates.
    alpha : float
        Two-sided level, in (0, 1).

    Returns
    -------
    result : dict
        Keys: estimate, se, se_corrected, crit_normal, crit_t, df,
        ci_lo, ci_hi, reject, reject_naive, n.

    References
    ----------
    Sant'Anna & Zhao (2020), J. Econometrics 219(1):101-122, eq. (2.6).
    MacKinnon & White (1985), J. Econometrics 29(3):305-325,
    doi:10.1016/0304-4076(85)90158-7.
    Bell & McCaffrey (2002), Survey Methodology 28(2):169-181.
    """
    yv = k.vec(y)
    dv = k.vec(D)
    n = len(yv)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    if len(dv) != n:
        raise ValueError("y and D must have the same length")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly between 0 and 1")
    n1 = sum(1 for v in dv if v >= 0.5)
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        raise ValueError("D must contain both treated and control units")
    fit = k.drdid_panel(yv, dv, X)
    nk = 1 + (k.ncol(k.mat(X)) if X is not None else 0)
    infl = math.sqrt(n / float(n - nk)) if n > nk else float("inf")
    se_c = fit["se"] * infl
    df = float(min(n1, n0) - 1)
    if df < 1.0:
        df = 1.0
    zc = k.qnorm(1.0 - alpha / 2.0)
    tc = _tquant(1.0 - alpha / 2.0, df)
    return RichResult(
        title="Size-corrected DR-DiD",
        summary_lines=[("df", df)],
        payload={
            "estimate": fit["tau"],
            "se": fit["se"],
            "se_corrected": se_c,
            "crit_normal": zc,
            "crit_t": tc,
            "df": df,
            "ci_lo": fit["tau"] - tc * se_c,
            "ci_hi": fit["tau"] + tc * se_c,
            "reject": 1.0 if abs(fit["tau"]) > tc * se_c else 0.0,
            "reject_naive": 1.0 if abs(fit["tau"]) > zc * fit["se"] else 0.0,
            "n": n,
            "method": "DR-DiD with finite-sample size correction",
        },
    )


def cheatsheet():
    return "drbsze: DR-DiD with finite-sample size correction"
