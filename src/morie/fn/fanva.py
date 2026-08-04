# morie.fn -- function file (rootcoder007/morie)
"""Functional analysis of variance.

Ramsay and Silverman (2005), *Functional Data Analysis*, 2nd ed.,
Springer, Chapter 13 "Modelling functional responses with multivariate
covariates", the one-way functional ANOVA: each curve is written as

    y_gi(t) = mu(t) + alpha_g(t) + eps_gi(t),

so at every argument value t the observations split into a grand mean
function, a treatment (group) effect function summing to zero over
groups, and a residual.  That is the decomposition named in the stub
docstring.

Because the model holds pointwise, the classical sums of squares hold
pointwise too, and the integrated versions are their integrals over
the whole interval.  When the curves are observed at a single argument
value the whole thing collapses to the textbook one-way ANOVA, which
is the anchor used for this module.

Group effects are the UNWEIGHTED deviations of the group mean
functions from the grand mean function of all curves.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["fanova"]


def _trapz(t, v):
    s = 0.0
    for i in range(len(t) - 1):
        s += 0.5 * (v[i] + v[i + 1]) * (t[i + 1] - t[i])
    return s


def fanova(functions, groups, t=None):
    """One-way functional ANOVA.

    Parameters
    ----------
    functions : array-like
        N-by-T matrix, one curve per row.
    groups : array-like
        N group labels.
    t : array-like, optional
        The argument grid of length T.  Defaults to equally spaced on
        [0, 1]; ignored (no integration is possible) when T == 1.

    Returns
    -------
    estimate : the integrated F ratio
    grand    : the grand mean function
    effects  : G-by-T matrix of group effect functions alpha_g
    ssb, ssw : pointwise between/within sums of squares
    ssb_int, ssw_int : their integrals
    F        : pointwise F ratio
    """
    Y = k.mat(functions)
    g = k.vec(groups)
    N = k.nrow(Y)
    if N == 0:
        raise ValueError("fanova: functions is empty")
    if len(g) != N:
        raise ValueError("fanova: groups must have one label per curve")
    T = k.ncol(Y)
    if T == 0:
        raise ValueError("fanova: functions has no argument values")
    levels = sorted(set(g))
    G = len(levels)
    if G < 2:
        raise ValueError("fanova: need at least two groups")
    if N <= G:
        raise ValueError("fanova: need more curves than groups")
    grand = [0.0] * T
    for i in range(N):
        for j in range(T):
            grand[j] += Y[i][j]
    grand = [v / float(N) for v in grand]
    gmeans = []
    counts = []
    for lv in levels:
        idx = [i for i in range(N) if g[i] == lv]
        counts.append(len(idx))
        m = [0.0] * T
        for i in idx:
            for j in range(T):
                m[j] += Y[i][j]
        gmeans.append([v / float(len(idx)) for v in m])
    effects = [[gmeans[a][j] - grand[j] for j in range(T)] for a in range(G)]
    ssb = [0.0] * T
    for a in range(G):
        for j in range(T):
            ssb[j] += counts[a] * effects[a][j] * effects[a][j]
    ssw = [0.0] * T
    for a in range(G):
        lv = levels[a]
        for i in range(N):
            if g[i] != lv:
                continue
            for j in range(T):
                r = Y[i][j] - gmeans[a][j]
                ssw[j] += r * r
    df1 = float(G - 1)
    df2 = float(N - G)
    Fp = []
    for j in range(T):
        Fp.append((ssb[j] / df1) / (ssw[j] / df2) if ssw[j] > 0.0 else float("inf"))
    if T == 1:
        tt = [0.0]
        ssb_int = ssb[0]
        ssw_int = ssw[0]
    else:
        tt = [i / float(T - 1) for i in range(T)] if t is None else k.vec(t)
        if len(tt) != T:
            raise ValueError("fanova: t must match the number of argument values")
        ssb_int = _trapz(tt, ssb)
        ssw_int = _trapz(tt, ssw)
    Fint = (ssb_int / df1) / (ssw_int / df2) if ssw_int > 0.0 else float("inf")
    return RichResult(
        title="Functional ANOVA",
        summary_lines=[("curves", N), ("groups", G), ("argument values", T), ("integrated F", Fint)],
        payload={
            "estimate": Fint,
            "grand": grand,
            "effects": effects,
            "ssb": ssb,
            "ssw": ssw,
            "ssb_int": ssb_int,
            "ssw_int": ssw_int,
            "F": Fp,
            "df1": df1,
            "df2": df2,
            "n": N,
            "method": "Ramsay-Silverman (2005) Ch.13 one-way functional ANOVA, pointwise decomposition integrated over the whole interval",
        },
    )


def cheatsheet():
    return "fanva: one-way functional ANOVA"
