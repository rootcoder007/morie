"""Cochran's Q heterogeneity statistic and the quantities read off it."""

from . import _array_core as np
from . import _stats_core as stats
from ._richresult import hypothesis_test_result

__all__ = ["ma_cochran_q"]


def ma_cochran_q(yi, vi):
    r"""Cochran's Q for between-study heterogeneity in a meta-analysis.

    With inverse-variance weights :math:`w_i = 1/v_i` and the
    fixed-effect summary :math:`\hat\theta_{FE}=\sum w_iy_i/\sum w_i`,

    .. math:: Q=\sum_i w_i\,(y_i-\hat\theta_{FE})^2 \;\sim\;\chi^2_{k-1}

    under homogeneity. Because :math:`E[Q]=k-1` when the studies share
    one true effect, the excess :math:`Q-(k-1)` is the raw material for
    everything else reported here:

    - :math:`\tau^2 = \max\{0,\;(Q-(k-1))/C\}` with
      :math:`C=\sum w_i-\sum w_i^2/\sum w_i` -- the DerSimonian-Laird
      moment estimator of between-study variance;
    - :math:`I^2=\max\{0,\;(Q-(k-1))/Q\}`, the share of total variation
      that is between-study rather than sampling;
    - :math:`H^2=Q/(k-1)`.

    Q has poor power with few studies and becomes trigger-happy with
    many, so a non-significant Q is weak evidence *for* homogeneity.
    :math:`I^2` is reported alongside because it does not grow with
    ``k`` the way Q does.

    Parameters
    ----------
    yi : array-like
        Observed effect sizes, one per study.
    vi : array-like
        Their sampling variances (not standard errors).

    Returns
    -------
    RichResult
        Keys ``statistic`` (Q), ``pvalue``, ``df``, ``theta_fe``,
        ``se_fe``, ``tau2``, ``i2``, ``h2``, ``k``, ``weights``.

    References
    ----------
    Cochran, W. G. (1954). The combination of estimates from different
    experiments. *Biometrics*, 10(1), 101-129.
    DerSimonian, R. & Laird, N. (1986). Meta-analysis in clinical
    trials. *Controlled Clinical Trials*, 7(3), 177-188.
    Higgins, J. P. T. & Thompson, S. G. (2002). Quantifying
    heterogeneity in a meta-analysis. *Statistics in Medicine*,
    21(11), 1539-1558.
    """
    y = [float(t) for t in np.asarray(yi, dtype=float).ravel().tolist()]
    v = [float(t) for t in np.asarray(vi, dtype=float).ravel().tolist()]
    if len(y) != len(v):
        raise ValueError("yi and vi must have the same length.")
    k = len(y)
    if k < 2:
        raise ValueError("need at least 2 studies.")
    if any(t <= 0 for t in v):
        raise ValueError("sampling variances must be strictly positive.")
    w = [1.0 / t for t in v]
    sw = sum(w)
    theta = sum(w[i] * y[i] for i in range(k)) / sw
    q = sum(w[i] * (y[i] - theta) ** 2 for i in range(k))
    df = k - 1
    c = sw - sum(t * t for t in w) / sw
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    return hypothesis_test_result(
        test_name="Cochran's Q test for heterogeneity",
        statistic=float(q),
        pvalue=float(stats.chi2.sf(q, df)),
        extra_summary=[("k", k), ("df", df), ("i2", i2)],
        extra_payload={
            "df": df,
            "k": k,
            "theta_fe": float(theta),
            "se_fe": float((1.0 / sw) ** 0.5),
            "tau2": float(tau2),
            "i2": float(i2),
            "h2": float(q / df),
            "weights": w,
            "c_constant": float(c),
            "method": "Cochran (1954) Q; DerSimonian-Laird tau^2; Higgins I^2",
        },
    )


def cheatsheet():
    return "macn: Cochran's Q heterogeneity test, with tau^2, I^2 and H^2"
