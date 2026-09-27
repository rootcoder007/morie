"""Correlation between and within subjects for repeated measurements."""

import math

from ._richresult import RichResult
from ._stats_core import f as fdist
from .nnlsq import _lstsq

__all__ = ["repeated_measures_correlation"]


def repeated_measures_correlation(x, y, subject):
    r"""Between- and within-subject correlation (Bland and Altman 1995).

    Hedderich, Sachs & Reynarowych (2023, Sec. 7.8.1.2, eqs 7.402-7.403):

    * between subjects: the correlation of the subject means, weighted by
      the number of measurements :math:`m_i` (7.402);
    * within subjects: from the analysis of covariance of y on subject and
      x, :math:`r = \mathrm{sign}(\hat\beta_x)\sqrt{SS_x/(SS_x + SS_{resid})}`
      (7.403), tested by the F test of x on 1 and N - k - 1 degrees of
      freedom. This is the repeated-measures correlation of Bakdash and
      Marusich (2017), ``rmcorr``.

    Parameters
    ----------
    x, y : sequence of float
    subject : sequence
        Subject (case) identifier of each measurement.

    Returns
    -------
    RichResult
        ``r_between``, ``r_within``, ``p_within``, ``df_within``,
        ``n_subjects``.

    References
    ----------
    Bland, J. M. & Altman, D. G. (1995). Calculating correlation
    coefficients with repeated observations. BMJ 310, 446 and 633.
    Bakdash, J. Z. & Marusich, L. R. (2017). Repeated measures correlation.
    Frontiers in Psychology 8, 456. Hedderich, J., Sachs, L. &
    Reynarowych, Z. (2023). Applied Statistics: Methods Using R. Springer,
    eqs (7.402)-(7.403).
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    sub = list(subject)
    N = len(xs)
    if not (len(ys) == len(sub) == N):
        raise ValueError("`x`, `y` and `subject` must be the same length")
    levels = list(dict.fromkeys(sub))
    k = len(levels)
    if k < 3 or N - k - 1 < 1:
        raise ValueError("need at least three subjects and N > k + 1")
    grp = {s: [i for i in range(N) if sub[i] == s] for s in levels}
    m = [len(grp[s]) for s in levels]
    mx = [sum(xs[i] for i in grp[s]) / len(grp[s]) for s in levels]
    my = [sum(ys[i] for i in grp[s]) / len(grp[s]) for s in levels]
    M = sum(m)
    sxy = (
        sum(a * b * c for a, b, c in zip(m, mx, my))
        - sum(a * b for a, b in zip(m, mx)) * sum(a * b for a, b in zip(m, my)) / M
    )
    sxx = sum(a * b * b for a, b in zip(m, mx)) - sum(a * b for a, b in zip(m, mx)) ** 2 / M
    syy = sum(a * b * b for a, b in zip(m, my)) - sum(a * b for a, b in zip(m, my)) ** 2 / M
    rb = sxy / math.sqrt(sxx * syy)
    dummies = [[1.0 if sub[i] == s else 0.0 for i in range(N)] for s in levels]
    rss_sub = sum((ys[i] - my[levels.index(sub[i])]) ** 2 for i in range(N))
    beta = _lstsq(dummies + [xs], ys)
    fit = [sum(c[i] * b for c, b in zip(dummies + [xs], beta)) for i in range(N)]
    rss_full = sum((a - b) ** 2 for a, b in zip(ys, fit))
    ssx = rss_sub - rss_full
    rw = math.copysign(math.sqrt(ssx / (ssx + rss_full)), beta[-1])
    df = N - k - 1
    fstat = ssx / (rss_full / df)
    return RichResult(
        title="Repeated-measures correlation",
        summary_lines=[("r between", rb), ("r within", rw), ("p within", float(fdist.sf(fstat, 1, df)))],
        payload={
            "r_between": rb,
            "r_within": rw,
            "p_within": float(fdist.sf(fstat, 1, df)),
            "df_within": df,
            "n_subjects": k,
        },
    )


def cheatsheet():
    return "rpmcor: Bland-Altman between-subject (weighted means) and within-subject (ANCOVA) correlation"
