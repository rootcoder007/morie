# morie.fn -- function file (rootcoder007/morie)
"""Research P12: ecological versus individual correlation.

Python twin of ``R/ecological.R`` (``morie_ecological_decompose``); the
identities are machine-checked in ``research/lean/P12Ecological.lean``
(Freedman, Pisani & Purves, Statistics 4e, ch. 9 sec. 4; Robinson 1950):

- ``Research.P12.within_orth``: the within-group residual is orthogonal to
  every group-level function
- ``Research.P12.cov_decomp``: cov(x, y) = cov(between) + cov(within)
- ``Research.P12.var_decomp``: var(x) = var(between) + var(within)
- ``Research.P12.ecological_ge``: zero within covariance implies
  corr(x, y)^2 <= corr(group means)^2
"""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = ["ecological_decompose"]


def _mean(v):
    return math.fsum(v) / len(v)


def _pcov(a, b):
    return math.fsum(u * v for u, v in zip(a, b)) / len(a) - _mean(a) * _mean(b)


def _sign(v):
    return (v > 0) - (v < 0)


def ecological_decompose(x, y, group):
    """Between/within decomposition of a correlation across groups.

    Splits each variable into its group mean (between part) and the
    residual (within part). Covariances and variances decompose exactly;
    when the within-group covariance is zero the group-level (ecological)
    correlation is at least as large in magnitude as the individual one.
    Otherwise the ecological correlation can be smaller or of the opposite
    sign (Robinson 1950). Population (divide by n) conventions throughout.

    Parameters
    ----------
    x, y : sequence of float
        One entry per individual.
    group : sequence
        Group label per individual.

    Returns
    -------
    RichResult
        ``cov``, ``var_x``, ``var_y`` (dicts with ``individual``,
        ``between``, ``within``), ``corr_individual``, ``corr_ecological``
        (correlation of the group means, weighted by group size),
        ``within_share_of_cov``, ``bound_applies``, ``sign_reversed`` and
        ``theorems``.

    Examples
    --------
    >>> d = ecological_decompose([0, 2, 1, 3], [1, 3, 0, 2], ["a", "a", "b", "b"])
    >>> round(d.corr_individual, 12), round(d.corr_ecological, 12), d.sign_reversed
    (0.6, -1.0, True)
    >>> {k: round(v, 12) for k, v in d.cov.items()}
    {'individual': 0.75, 'between': -0.25, 'within': 1.0}
    """
    n = len(x)
    if len(y) != n or len(group) != n:
        raise ValueError("x, y and group must have equal length")
    if n < 2:
        raise ValueError("need at least two individuals")
    x = [float(v) for v in x]
    y = [float(v) for v in y]
    g = [str(v) for v in group]
    idx = {}
    for i, k in enumerate(g):
        idx.setdefault(k, []).append(i)
    mx = {k: _mean([x[i] for i in ii]) for k, ii in idx.items()}
    my = {k: _mean([y[i] for i in ii]) for k, ii in idx.items()}
    bx = [mx[k] for k in g]
    by = [my[k] for k in g]
    wx = [a - b for a, b in zip(x, bx)]
    wy = [a - b for a, b in zip(y, by)]

    def split(a, b, c, d, e, f):
        return {"individual": _pcov(a, b), "between": _pcov(c, d), "within": _pcov(e, f)}

    cv = split(x, y, bx, by, wx, wy)
    vx = split(x, x, bx, bx, wx, wx)
    vy = split(y, y, by, by, wy, wy)
    ci = (
        cv["individual"] / math.sqrt(vx["individual"] * vy["individual"])
        if vx["individual"] > 0 and vy["individual"] > 0
        else math.nan
    )
    ce = (
        cv["between"] / math.sqrt(vx["between"] * vy["between"])
        if vx["between"] > 0 and vy["between"] > 0
        else math.nan
    )
    rev = not (math.isnan(ci) or math.isnan(ce)) and _sign(ci) != _sign(ce) and ci != 0 and ce != 0
    return RichResult(
        title="Ecological versus individual correlation",
        payload={
            "cov": cv,
            "var_x": vx,
            "var_y": vy,
            "corr_individual": ci,
            "corr_ecological": ce,
            "within_share_of_cov": cv["within"] / cv["individual"] if cv["individual"] != 0 else math.nan,
            "bound_applies": abs(cv["within"]) < 1e-12 * max(1.0, abs(cv["individual"])),
            "sign_reversed": rev,
            "theorems": [
                "Research.P12.within_orth",
                "Research.P12.cov_decomp",
                "Research.P12.var_decomp",
                "Research.P12.ecological_ge",
            ],
        },
    )


def cheatsheet() -> str:
    return (
        "ecological_decompose(x, y, group) -> between/within split of cov and var, individual vs "
        "ecological correlation, Robinson sign reversal (Research P12)."
    )
