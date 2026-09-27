"""Chi-square test of independence with a Monte Carlo p-value over tables with fixed margins."""

import random

from ._richresult import RichResult

__all__ = ["chisq_monte_carlo"]


def chisq_monte_carlo(table, B=2000, seed=None):
    r"""Pearson's :math:`X^2` for an r x c table with a p-value simulated under fixed margins.

    Tables are drawn from the conditional (multivariate hypergeometric)
    distribution given both margins by randomly permuting the column labels
    of the n units, the distribution ``r2dtable`` samples; the p-value is
    :math:`(1 + \#\{X^2_b \ge X^2\})/(B + 1)` as ``chisq.test(simulate.p.value
    = TRUE)`` (Hedderich, Sachs & Reynarowych 2023, p. 739). Draws use
    Python's ``random.Random(seed)``, so they differ from R's stream.

    Parameters
    ----------
    table : r x c nested sequence of counts
    B : int
        Number of simulated tables.
    seed : int, optional

    Returns
    -------
    RichResult
        ``statistic``, ``p_value``, ``B``.

    References
    ----------
    Hope, A. C. A. (1968). A simplified Monte Carlo significance test
    procedure. JRSS B 30, 582-598.
    """
    t = [[int(v) for v in row] for row in table]
    nr, nc = len(t), len(t[0])
    if nr < 2 or nc < 2 or any(len(row) != nc for row in t) or any(v < 0 for row in t for v in row):
        raise ValueError("need an r x c table (r, c >= 2) of non-negative counts")
    rs = [sum(row) for row in t]
    cs = [sum(t[i][j] for i in range(nr)) for j in range(nc)]
    n = sum(rs)
    if min(rs) == 0 or min(cs) == 0:
        raise ValueError("a margin is zero")
    e = [[rs[i] * cs[j] / n for j in range(nc)] for i in range(nr)]

    def stat(tab):
        return sum((tab[i][j] - e[i][j]) ** 2 / e[i][j] for i in range(nr) for j in range(nc))

    x = stat(t)
    rowlab = [i for i in range(nr) for _ in range(rs[i])]
    collab = [j for j in range(nc) for _ in range(cs[j])]
    rng = random.Random(seed)
    almost = 1 - 64 * 2.220446049250313e-16
    hits = 0
    for _ in range(int(B)):
        rng.shuffle(collab)
        tab = [[0] * nc for _ in range(nr)]
        for i, j in zip(rowlab, collab):
            tab[i][j] += 1
        hits += stat(tab) >= almost * x
    p = (1 + hits) / (B + 1)
    return RichResult(
        title="Chi-square test, Monte Carlo p-value",
        summary_lines=[("statistic", x), ("p_value", p)],
        payload={"statistic": x, "p_value": p, "B": int(B)},
    )


def cheatsheet():
    return "chisqmc: p = (1 + #{X2_b >= X2}) / (B + 1) over tables with the observed margins"
