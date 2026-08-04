# morie.fn -- function file (rootcoder007/morie)
"""Treatments-versus-control precedence test for the tree alternative."""

import math

from ._richresult import RichResult

__all__ = ['ctrltree', 'gibbons_fligner_wolfe_test']


def ctrltree(samples, r=None):
    """Chakraborti-Desu W: treatment observations preceding T.

    Section 10.7.2 (book p. 373), eq. (10.7.3).  Sample 1 is the
    control and T is one of its order statistics (the median by
    default).  W counts the observations in treatment groups
    2, ..., k that precede T, and H0 is rejected in favour of the tree
    alternative for small W.  The book shows that, because W is the
    total precedence count, its null distribution is that of the
    two-sample precedence statistic with sample sizes n_1 and N - n_1,
    i.e. the placement law of Problem 2.28(c):

    .. math:: P[W = w] = \\frac{\\binom{n_1 + M - r - w}{M - w}
        \\binom{r + w - 1}{w}}{\\binom{n_1 + M}{M}},
        \\qquad M = N - n_1.

    NOTE ON THE MODULE LABEL: the generated stub called this a
    "Fligner-Wolfe test".  Fligner & Killeen and the Fligner-Wolfe
    treatments-versus-control test are not in Gibbons & Chakraborti
    (2011); the only Fligner reference the book carries is Fligner and
    Wolfe (1976) on placements (pp. 65, 70).  What the cited source
    gives for this problem is the Chakraborti-Desu test, implemented
    here.

    Parameters
    ----------
    samples : sequence of sequence of float
        The k samples; ``samples[0]`` is the control.
    r : int, optional
        Index of the control order statistic T (defaults to the
        control median index [n_1 / 2] + 1).

    Returns
    -------
    RichResult
        keys ``statistic`` (W), ``p_value`` (lower tail), ``pmf``,
        ``t`` (the control order statistic used), ``r``, ``mtreat``
        (N - n_1), ``mean``, ``k``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 10.7.2, eq. (10.7.3), p. 373
    (Chakraborti and Desu, 1988b); null law Problem 2.28(c), p. 70.
    """
    ss = [[float(v) for v in s] for s in samples]
    k = len(ss)
    if k < 2:
        raise ValueError("need at least 2 samples.")
    ctrl = sorted(ss[0])
    n1 = len(ctrl)
    if n1 < 1:
        raise ValueError("the control sample must be non-empty.")
    rr = (n1 // 2) + 1 if r is None else int(r)
    if not 1 <= rr <= n1:
        raise ValueError("r must lie in 1..n1.")
    t = ctrl[rr - 1]
    treat = [v for s in ss[1:] for v in s]
    mt = len(treat)
    if mt < 1:
        raise ValueError("need at least one treatment observation.")
    w = sum(1 for v in treat if v < t)
    den = math.comb(n1 + mt, mt)
    pmf = [
        math.comb(n1 + mt - rr - j, mt - j) * math.comb(rr + j - 1, j) / den
        for j in range(mt + 1)
    ]
    mean = sum(j * p for j, p in enumerate(pmf))
    return RichResult(
        payload={
            "statistic": int(w),
            "p_value": float(min(1.0, sum(pmf[: w + 1]))),
            "pmf": pmf,
            "t": float(t),
            "r": int(rr),
            "mtreat": int(mt),
            "mean": float(mean),
            "k": int(k),
            "method": "treatments-vs-control precedence test, eq. (10.7.3)",
        }
    )


gibbons_fligner_wolfe_test = ctrltree
