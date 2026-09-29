# morie.fn -- function file from book-equation translation pipeline (rootcoder007/morie)
"""James-Stein shrinkage estimator for a multivariate normal mean."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def james_stein(
    x,
    *,
    target: float | None = None,
    sigma2: float = 1.0,
) -> DescriptiveResult:
    r"""James-Stein shrinkage estimator for a multivariate normal mean.

    For :math:`X_i \sim N(\theta_i, \sigma^2)`, :math:`i = 1..p`, the
    positive-part estimator shrinks every coordinate toward a common target
    :math:`t`:

    .. math::

        \hat\theta_i = t + c\,(X_i - t), \qquad
        c = \max\Big(0,\; 1 - \frac{k\,\sigma^2}{\sum_i (X_i - t)^2}\Big).

    With a fixed target :math:`k = p - 2` (James and Stein 1961). When the
    target is the grand mean :math:`\bar X`, estimated from the same data,
    one degree of freedom is spent on it and :math:`k = p - 3` (Efron and
    Morris 1973, 1975); using :math:`p - 2` there over-shrinks.

    Parameters
    ----------
    x : array-like
        Observed means (length >= 3).
    target : float or None
        Fixed shrinkage target; the grand mean (with ``p - 3``) if None.
    sigma2 : float
        Known sampling variance of each coordinate (default 1).

    Returns
    -------
    DescriptiveResult
        ``value`` = shrinkage factor ``c`` (0 = full shrinkage, 1 = none);
        ``extra`` has ``js_estimates``, ``target``, ``k``, ``sigma2`` and
        the rest.

    References
    ----------
    James, W. and Stein, C. (1961). Estimation with quadratic loss.
    Proc. Fourth Berkeley Symp. 1, 361-379.
    Efron, B. and Morris, C. (1973). Stein's estimation rule and its
    competitors -- an empirical Bayes approach. JASA 68, 117-130.
    Efron, B. and Morris, C. (1975). Data analysis using Stein's estimator
    and its generalizations. JASA 70, 311-319.

    Examples
    --------
    >>> r = james_stein([10.0, -5.0, 3.0, 0.1, -2.0])
    >>> round(r.value, 12), r.extra["k"]
    (0.984682311133, 2)
    """
    xs = [float(v) for v in (x.tolist() if hasattr(x, "tolist") else x)]
    p = len(xs)
    if p < 3:
        raise ValueError("James-Stein requires >= 3 means (Stein's paradox)")
    s2 = float(sigma2)
    if not s2 > 0:
        raise ValueError("sigma2 must be positive")
    if target is None:
        tgt = math.fsum(xs) / p
        k = p - 3
    else:
        tgt = float(target)
        k = p - 2
    diff = [v - tgt for v in xs]
    ss = math.fsum(d * d for d in diff)
    shrinkage = 0.0 if ss < 1e-30 else max(0.0, 1.0 - k * s2 / ss)
    js = [tgt + shrinkage * d for d in diff]
    return DescriptiveResult(
        name="James-Stein shrinkage estimator",
        value=float(shrinkage),
        extra={
            "p": p,
            "target": tgt,
            "k": k,
            "sigma2": s2,
            "shrinkage_factor": shrinkage,
            "js_estimates": js,
            "original_means": xs,
            "mse_reduction_bound": round(1 - shrinkage**2, 4) if shrinkage < 1 else 0.0,
        },
    )


jamste = james_stein


def cheatsheet() -> str:
    return "james_stein({}) -> James-Stein shrinkage estimator."


# compact alias per ledger/NAMING.md
jamesstein = james_stein
