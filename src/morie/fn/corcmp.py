"""Comparison and pooling of independent correlation coefficients."""

import math

from ._richresult import RichResult
from ._stats_core import chi2, norm
from ._stats_core import t as tdist

__all__ = ["compare_correlations"]


def compare_correlations(r, n, rho0=None, confidence=0.95):
    r"""Homogeneity test and pooled estimate for k independent correlations.

    Hedderich, Sachs & Reynarowych (2023, Sec. 7.8.1.1 and 7.8.1.4), with
    :math:`\dot z_i = \tanh^{-1} r_i` and weights :math:`n_i - 3`:

    * pooled :math:`\bar z = \sum(n_i-3)\dot z_i/\sum(n_i-3)` with standard
      error :math:`1/\sqrt{\sum(n_i-3)}` and its back-transformed interval
      (7.401), (7.407)-(7.408);
    * :math:`\hat\chi^2 = \sum(n_i-3)(\dot z_i - \bar z)^2` on :math:`k-1`
      degrees of freedom (7.409), or against a given :math:`\rho_0` on k
      (7.406);
    * for two samples :math:`\hat z = |\dot z_1 - \dot z_2|/
      \sqrt{1/(n_1-3) + 1/(n_2-3)}` (7.400);
    * the weighted mean :math:`r_{gem} = \sum(n_i-1)r_i/\sum(n_i-1)` and
      :math:`\hat t = r_{gem}\sqrt{(n-k-1)/(1-r_{gem}^2)}`,
      :math:`n = \sum n_i` (7.395)-(7.396).

    Parameters
    ----------
    r : sequence of float
        Correlations, at least two.
    n : sequence of int
        Their sample sizes, each at least 4.
    rho0 : float, optional
        Hypothesised common correlation.
    confidence : float

    Returns
    -------
    RichResult
        ``r_pooled``, ``ci``, ``z_pooled``, ``se_pooled``, ``chi2``,
        ``df``, ``p_value``, and ``chi2_rho0``/``p_rho0`` when ``rho0`` is
        given, ``z_two`` and ``p_two`` for two samples, ``r_gem``,
        ``t_gem``, ``p_gem``.

    References
    ----------
    Hedderich, J., Sachs, L. & Reynarowych, Z. (2023). Applied Statistics:
    Methods Using R. Springer, eqs (7.395)-(7.409).
    """
    rr = [float(v) for v in r]
    nn = [int(v) for v in n]
    k = len(rr)
    if k < 2 or len(nn) != k or any(not -1 < v < 1 for v in rr) or any(v < 4 for v in nn):
        raise ValueError("need at least two correlations in (-1, 1) with n_i >= 4")
    z = [math.atanh(v) for v in rr]
    w = [v - 3 for v in nn]
    sw = sum(w)
    zb = sum(a * b for a, b in zip(w, z)) / sw
    se = 1.0 / math.sqrt(sw)
    q = float(norm.ppf(1 - (1 - confidence) / 2))
    x2 = sum(a * (b - zb) ** 2 for a, b in zip(w, z))
    out = {
        "r_pooled": math.tanh(zb),
        "ci": (math.tanh(zb - q * se), math.tanh(zb + q * se)),
        "z_pooled": zb,
        "se_pooled": se,
        "chi2": x2,
        "df": k - 1,
        "p_value": float(chi2.sf(x2, k - 1)),
    }
    if rho0 is not None:
        z0 = math.atanh(float(rho0))
        x2r = sum(a * (b - z0) ** 2 for a, b in zip(w, z))
        out.update({"chi2_rho0": x2r, "p_rho0": float(chi2.sf(x2r, k))})
    if k == 2:
        zz = abs(z[0] - z[1]) / math.sqrt(1 / w[0] + 1 / w[1])
        out.update({"z_two": zz, "p_two": float(2 * norm.sf(zz))})
    ntot = sum(nn)
    rg = sum((a - 1) * b for a, b in zip(nn, rr)) / sum(a - 1 for a in nn)
    tg = rg * math.sqrt((ntot - k - 1) / (1 - rg * rg))
    out.update({"r_gem": rg, "t_gem": tg, "p_gem": float(2 * tdist.sf(abs(tg), ntot - k - 1))})
    return RichResult(
        title="Comparison of independent correlations",
        summary_lines=[("k", k), ("pooled r", out["r_pooled"]), ("chi2", x2)],
        payload=out,
    )


def cheatsheet():
    return "corcmp: Fisher-z comparison, homogeneity chi2 and pooled r of independent correlations"
