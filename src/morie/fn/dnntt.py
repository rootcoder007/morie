# morie.fn -- function file (rootcoder007/morie)
"""Dunnett's test -- multiple treatment groups vs control."""

from __future__ import annotations

import math

from . import _array_core as np
from ._containers import DescriptiveResult
from ._schab_st import gauss_legendre

_GL16 = None


def _gl16():
    global _GL16
    if _GL16 is None:
        nodes, weights = gauss_legendre(16)
        _GL16 = ([float(v) for v in nodes], [float(v) for v in weights])
    return _GL16


def _panels(a, b, m):
    t, w = _gl16()
    h = (b - a) / m
    pts, wts = [], []
    for j in range(m):
        lo = a + j * h
        pts.extend(lo + 0.5 * h * (v + 1.0) for v in t)
        wts.extend(0.5 * h * v for v in w)
    return pts, wts


def _Phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _dunnett_cdf(c, lam, df):
    """P(max_j |T_j| <= c) for Dunnett's multivariate t, correlations lam_i lam_j."""
    zs, zw = _panels(-9.0, 9.0, 36)
    phi = [math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) for z in zs]
    sq = [math.sqrt(1.0 - lj * lj) for lj in lam]

    def inner(cs):
        tot = 0.0
        for z, wz, pz in zip(zs, zw, phi):
            prod = 1.0
            for lj, q in zip(lam, sq):
                prod *= _Phi((lj * z + cs) / q) - _Phi((lj * z - cs) / q)
            tot += wz * pz * prod
        return tot

    if df is None or math.isinf(df):
        return inner(c)
    upper = math.sqrt(160.0 / df) + 1.0
    ss, sw = _panels(0.0, upper, 48)
    logc = 0.5 * df * math.log(df) - math.lgamma(0.5 * df) - (0.5 * df - 1.0) * math.log(2.0)
    tot = 0.0
    for s, w in zip(ss, sw):
        if s <= 0.0:
            continue
        dens = math.exp(logc + (df - 1.0) * math.log(s) - 0.5 * df * s * s)
        if dens < 1e-300:
            continue
        tot += w * dens * inner(c * s)
    return tot


def dunnett_test(
    control: np.ndarray,
    *treatment_groups: np.ndarray,
) -> DescriptiveResult:
    """Dunnett's many-to-one comparisons with exact multivariate-t p-values.

    Each treatment mean is compared with the control, :math:`t_i = (\\bar y_i
    - \\bar y_0)/\\sqrt{\\mathrm{MSE}(1/n_i + 1/n_0)}` on the pooled
    within-group mean square with :math:`N - k - 1` degrees of freedom.
    Jointly the :math:`t_i` follow a multivariate t whose correlations are
    :math:`\\lambda_i\\lambda_j`, :math:`\\lambda_i = \\sqrt{n_i/(n_i + n_0)}`
    (Dunnett 1955), and the two-sided adjusted p-value of comparison i is
    :math:`1 - P(\\max_j |T_j| \\le |t_i|)`, computed by quadrature of
    Dunnett's double integral over the standard normal and the
    :math:`\\sqrt{\\chi^2_\\nu/\\nu}` scale. This is multcomp's Dunnett
    contrast test (``glht(..., mcp(g = "Dunnett"))``).

    Parameters
    ----------
    control : array-like
    *treatment_groups : array-like

    Returns
    -------
    DescriptiveResult
        ``extra["comparisons"]`` lists ``group``, ``diff``, ``se``, ``t`` and
        ``p_adj`` (two-sided, Dunnett-adjusted) for each treatment; ``mse``,
        ``df``, ``n_control``.

    References
    ----------
    Dunnett, C. W. (1955). A multiple comparison procedure for comparing
    several treatments with a control. JASA 50, 1096-1121. Hothorn, T.,
    Bretz, F. & Westfall, P. (2008). Simultaneous inference in general
    parametric models. Biometrical Journal 50, 346-363.
    """
    if len(treatment_groups) < 1:
        raise ValueError("Need >= 1 treatment group.")
    ctrl = [float(v) for v in np.asarray(control, dtype=float).ravel()]
    groups = [[float(v) for v in np.asarray(g, dtype=float).ravel()] for g in treatment_groups]
    k = len(groups)
    n0 = len(ctrl)
    N = n0 + sum(len(g) for g in groups)
    df = N - k - 1
    if df < 1:
        raise ValueError("Need more observations than groups.")
    m0 = sum(ctrl) / n0
    ss = sum((v - m0) ** 2 for v in ctrl)
    for g in groups:
        mg = sum(g) / len(g)
        ss += sum((v - mg) ** 2 for v in g)
    mse = ss / df
    lam = [math.sqrt(len(g) / (len(g) + n0)) for g in groups]
    results = []
    for i, g in enumerate(groups):
        diff = sum(g) / len(g) - m0
        se = math.sqrt(mse * (1.0 / len(g) + 1.0 / n0))
        t_stat = diff / se if se > 0 else 0.0
        p_adj = max(0.0, min(1.0, 1.0 - _dunnett_cdf(abs(t_stat), lam, df)))
        results.append(
            {"group": i + 1, "diff": float(diff), "se": float(se), "t": float(t_stat), "p_adj": float(p_adj)}
        )
    return DescriptiveResult(
        name="dunnett",
        value=k,
        extra={
            "comparisons": results,
            "mse": float(mse),
            "df": df,
            "n_control": n0,
            "correction": "dunnett (multivariate t)",
        },
    )


dnntt = dunnett_test


def cheatsheet() -> str:
    return "dunnett_test({}) -> Dunnett's test -- multiple treatment groups vs control."


# compact alias per ledger/NAMING.md
dunnetttest = dunnett_test
