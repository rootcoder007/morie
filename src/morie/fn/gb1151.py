# morie.fn -- function file (rootcoder007/morie)
"""Fieller-Hartley-Pearson normal-scores correlation coefficient."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['normcorr', 'gibbons_blomqvist_q']


def _enosc(i, n, lo=-8.0, hi=8.0, nodes=4001):
    """E[Z_(i:n)] for the standard normal, fixed-grid Simpson."""
    if nodes % 2 == 0:
        nodes += 1
    h = (hi - lo) / (nodes - 1)
    coef = math.exp(
        math.lgamma(n + 1.0) - math.lgamma(i) - math.lgamma(n - i + 1.0)
    )
    total = 0.0
    for k in range(nodes):
        z = lo + k * h
        w = 1.0 if k in (0, nodes - 1) else (4.0 if k % 2 else 2.0)
        p = stats.norm.cdf(z)
        total += w * z * p ** (i - 1) * (1.0 - p) ** (n - i) * stats.norm.pdf(z)
    return coef * total * h / 3.0


def normcorr(x, y, rho=0.0, nodes=4001):
    """R_F, the correlation of the expected normal scores.

    Section 11.5 (book p. 422).  Replacing each rank by the expected
    normal order statistic xi_i = E(Z_(i)) and correlating the derived
    pairs gives

    .. math:: R_F = \\frac{\\sum_{i=1}^{n}\\xi_i \\xi_{s_i}}
        {\\sum_{i=1}^{n}\\xi_i^2},

    where s_i is the rank of the Y paired with the i-th smallest X.
    Fieller et al. (1957) show that Z_F = tanh^{-1} R_F is
    approximately normal with

    .. math:: E[Z_F] = \\tanh^{-1}\\rho
        \\left(1 - \\frac{0.6}{n+8}\\right), \\qquad
        Var[Z_F] = \\frac{1}{n-3}.

    NOTE ON THE MODULE LABEL: the generated stub called this
    "Blomqvist q (medial correlation)".  The name Blomqvist does not
    occur anywhere in Gibbons & Chakraborti (2011); Sec. 11.5, the
    book's only "another measure of association" for paired samples,
    is this Fieller-Hartley-Pearson coefficient, which is what this
    module implements.

    Parameters
    ----------
    x, y : sequence of float
        Paired observations, n >= 4.
    rho : float, optional
        Population correlation used for E[Z_F] (default 0).
    nodes : int, optional
        Simpson nodes for the expected order statistics (default 4001).

    Returns
    -------
    RichResult
        keys ``statistic`` (R_F), ``zf``, ``mean_zf``, ``var_zf``,
        ``z``, ``p_value``, ``scores``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 11.5, p. 422 (Fieller, Hartley
    and Pearson, 1957; Fieller and Pearson, 1961).
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    n = len(xs)
    if len(ys) != n:
        raise ValueError("x and y must have the same length.")
    if n < 4:
        raise ValueError("need at least 4 pairs.")
    xi = [_enosc(i, n, nodes=nodes) for i in range(1, n + 1)]
    ox = sorted(range(n), key=lambda i: xs[i])
    ry = [0] * n
    oy = sorted(range(n), key=lambda i: ys[i])
    for pos, idx in enumerate(oy):
        ry[idx] = pos + 1
    num = sum(xi[i] * xi[ry[ox[i]] - 1] for i in range(n))
    den = sum(v * v for v in xi)
    rf = num / den
    rf = min(1.0 - 1e-15, max(-1.0 + 1e-15, rf))
    zf = math.atanh(rf)
    mz = math.atanh(float(rho)) * (1.0 - 0.6 / (n + 8.0))
    vz = 1.0 / (n - 3.0)
    z = (zf - mz) / math.sqrt(vz)
    return RichResult(
        payload={
            "statistic": float(rf),
            "zf": float(zf),
            "mean_zf": float(mz),
            "var_zf": float(vz),
            "z": float(z),
            "p_value": float(2.0 * (1.0 - stats.norm.cdf(abs(z)))),
            "scores": xi,
            "n": n,
            "method": "Fieller-Hartley-Pearson normal-scores R_F (Sec. 11.5)",
        }
    )


gibbons_blomqvist_q = normcorr
