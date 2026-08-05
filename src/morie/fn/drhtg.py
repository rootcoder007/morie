# morie.fn -- function file (rootcoder007/morie)
"""Heterogeneous conditional ATT: the DR-DiD moment inside each stratum.

The doubly robust moment of Sant'Anna, P. H. C. and Zhao, J. (2020),
*Journal of Econometrics* 219(1), 101-122, eq. (2.6), is evaluated
separately on each stratum of a discrete covariate, giving

    CATT(x) = E[(w1(D) - w0(D, X; pi)) (dY - mu_0(X)) | S = x]

with the weights renormalised within the stratum, so every CATT(x) is a
self-contained DR estimate rather than a reweighting of a pooled fit.
Athey, S. and Imbens, G. (2016), *PNAS* 113(27), 7353-7360, is the
framing: the object of interest is the whole profile of subgroup
effects, and the pooled ATT is recovered as the stratum-size-weighted
average of the profile -- which is the identity this module asserts
back, so a single-stratum call must reproduce the pooled estimator
exactly.

The reported heterogeneity statistic is the size-weighted variance of
CATT across strata; it is exactly zero when all strata share an effect.
"""

from __future__ import annotations

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["dr_did_heterogeneity"]


def dr_did_heterogeneity(y, D, X=None, strata=None):
    """DR-DiD CATT within each stratum.

    Parameters
    ----------
    y : array-like
        Outcome change dY = Y_post - Y_pre, one entry per unit.
    D : array-like
        Binary treatment indicator.
    X : 2-D array-like, optional
        Baseline covariates used inside each stratum fit.
    strata : array-like, optional
        Discrete stratum label per unit; ``None`` is a single stratum.

    Returns
    -------
    result : dict
        Keys: estimate (size-weighted mean CATT), strata, catt, se,
        n_stratum, pooled, hetero_var, n.

    References
    ----------
    Sant'Anna & Zhao (2020), J. Econometrics 219(1):101-122, eq. (2.6),
    doi:10.1016/j.jeconom.2020.06.003.
    Athey & Imbens (2016), PNAS 113(27):7353-7360,
    doi:10.1073/pnas.1510489113.
    """
    yv = k.vec(y)
    dv = k.vec(D)
    n = len(yv)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    if len(dv) != n:
        raise ValueError("y and D must have the same length")
    Xr = k.mat(X) if X is not None else None
    st = [str(x) for x in strata] if strata is not None else ["all"] * n
    if len(st) != n:
        raise ValueError("strata must have the same length as y")
    labels = []
    for s in st:
        if s not in labels:
            labels.append(s)
    labels = sorted(labels)
    catt, se, ns = [], [], []
    for s in labels:
        idx = [i for i in range(n) if st[i] == s]
        ds = [dv[i] for i in idx]
        if len(idx) < 3 or sum(ds) <= 0.0 or sum(ds) >= len(ds):
            catt.append(float("nan"))
            se.append(float("nan"))
            ns.append(float(len(idx)))
            continue
        fit = k.drdid_panel([yv[i] for i in idx], ds,
                            [Xr[i] for i in idx] if Xr is not None else None)
        catt.append(fit["tau"])
        se.append(fit["se"])
        ns.append(float(len(idx)))
    num, den = 0.0, 0.0
    for j in range(len(labels)):
        if catt[j] == catt[j]:
            num += ns[j] * catt[j]
            den += ns[j]
    est = (num / den) if den > 0.0 else float("nan")
    hv = 0.0
    if den > 0.0:
        for j in range(len(labels)):
            if catt[j] == catt[j]:
                hv += ns[j] * (catt[j] - est) ** 2
        hv = hv / den
    pooled = k.drdid_panel(yv, dv, Xr)["tau"] \
        if (0.0 < sum(dv) < float(n)) else float("nan")
    return RichResult(
        title="DR-DiD heterogeneous CATT",
        summary_lines=[("strata", len(labels))],
        payload={
            "estimate": est,
            "strata": labels,
            "catt": catt,
            "se": se,
            "n_stratum": ns,
            "pooled": pooled,
            "hetero_var": hv,
            "n": n,
            "method": "DR-DiD heterogeneous CATT",
        },
    )


def cheatsheet():
    return "drhtg: DR-DiD heterogeneous CATT"
