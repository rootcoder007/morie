# morie.fn -- function file (rootcoder007/morie)
"""Peto's one-step pooled odds ratio."""

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ma_peto_or"]


def ma_peto_or(a, b, c, d, level=0.95):
    """Pool 2x2 tables when the events are rare enough to break the others.

    With a zero cell the ordinary log odds ratio is undefined and the
    usual repair -- adding 0.5 everywhere -- biases the pooled estimate.
    Peto's estimator never forms a ratio per study: it accumulates
    observed-minus-expected counts against their hypergeometric variance,
    so a zero cell contributes without special handling.  The price is a
    bias when the odds ratio is far from one or the groups are badly
    unbalanced.

    Formula: ``ln OR = sum(O_i - E_i)/sum(V_i)`` with ``O = a``,
    ``E = (a+b)(a+c)/N`` and ``V = (a+b)(c+d)(a+c)(b+d)/(N^2 (N-1))``;
    ``se = 1/sqrt(sum V)`` -- Peto et al. (1977), Appendix.

    Parameters
    ----------
    a, b, c, d : array-like
        Per-study cells: events and non-events in the treated arm, then in
        the control arm.
    level : float, default 0.95
        Confidence level.

    Returns
    -------
    RichResult
        ``OR``, ``log_OR``, ``se_log``, ``ci`` (lower, upper), ``O_E``,
        ``V``, ``k``.

    References
    ----------
    Peto, R., Pike, M. C., Armitage, P., Breslow, N. E., Cox, D. R.,
    Howard, S. V., Mantel, N., McPherson, K., Peto, J. and Smith, P. G.
    (1977).  Design and analysis of randomized clinical trials requiring
    prolonged observation of each patient.  II.  British Journal of Cancer
    35(1):1-39.  doi:10.1038/bjc.1977.1.
    """
    A = [float(t) for t in core.vec(a)]
    B = [float(t) for t in core.vec(b)]
    C = [float(t) for t in core.vec(c)]
    D = [float(t) for t in core.vec(d)]
    k = len(A)
    if k == 0:
        raise ValueError("no tables")
    if not (len(B) == len(C) == len(D) == k):
        raise ValueError("the four cell vectors must have equal length")
    if any(t < 0.0 for t in A + B + C + D):
        raise ValueError("cell counts must be non-negative")
    oe = 0.0
    vv = 0.0
    for i in range(k):
        n = A[i] + B[i] + C[i] + D[i]
        if n <= 1.0:
            raise ValueError("each table needs at least two observations")
        e = (A[i] + B[i]) * (A[i] + C[i]) / n
        v = ((A[i] + B[i]) * (C[i] + D[i]) * (A[i] + C[i]) * (B[i] + D[i])
             / (n * n * (n - 1.0)))
        oe += A[i] - e
        vv += v
    if vv <= 0.0:
        raise ValueError("the pooled variance is zero; no table is informative")
    lor = oe / vv
    se = 1.0 / math.sqrt(vv)
    z = core.qnorm(1.0 - (1.0 - float(level)) / 2.0)
    return RichResult(payload={
        "OR": math.exp(lor), "log_OR": lor, "se_log": se,
        "ci": [math.exp(lor - z * se), math.exp(lor + z * se)],
        "O_E": oe, "V": vv, "k": k,
        "method": "Peto one-step pooled odds ratio"})


def cheatsheet():
    return "mapeto: Peto's one-step pooled odds ratio for sparse 2x2 tables"


# compact alias per ledger/NAMING.md
mapetoor = ma_peto_or
