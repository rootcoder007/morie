"""Tukey-Kramer simultaneous intervals for all pairwise differences of means.

Wilcox (2017), Modern Statistics for the Social and Behavioral Sciences, eq (12.4).
"""

import math

from ._richresult import RichResult
from .ptukey import _ptukey
from .qtukey import qtukey

__all__ = ["tukeykramer"]


def tukeykramer(groups, alpha=0.05):
    """(Xbar_j - Xbar_k) +- q_{1-alpha}(J, nu) sqrt((MSWG/2)(1/n_j + 1/n_k)) (12.4).

    q is the studentized range quantile for J means and nu = N - J degrees of
    freedom, MSWG the mean square within groups; adjusted p-values come from
    the studentized range distribution of sqrt(2)|t|.

    Parameters
    ----------
    groups : list of sequences
    alpha : float

    Returns
    -------
    RichResult
        Keys: q, mswg, df, comparisons (j, k, diff, lower, upper, p_adj).

    References
    ----------
    Kramer, C. Y. (1956). Biometrics 12, 307-310.
    Wilcox, R. R. (2017). Modern Statistics for the Social and Behavioral
    Sciences (2nd ed.). CRC Press. Eq (12.4).

    Examples
    --------
    >>> r = tukeykramer([[1, 2, 3], [2, 3, 4], [6, 7, 8]])
    >>> r["df"], r["mswg"]
    (6, 1.0)
    """
    gs = [[float(v) for v in g] for g in groups]
    J = len(gs)
    N = sum(len(g) for g in gs)
    if J < 2 or any(len(g) < 1 for g in gs) or N - J < 2:
        raise ValueError("need at least two groups and N - J >= 2")
    means = [sum(g) / len(g) for g in gs]
    df = N - J
    mswg = sum(sum((v - m) ** 2 for v in g) for g, m in zip(gs, means)) / df
    q = qtukey(1 - alpha, J, df)["q"]
    comps = []
    for j in range(J):
        for k in range(j + 1, J):
            se = math.sqrt(mswg / 2 * (1 / len(gs[j]) + 1 / len(gs[k])))
            diff = means[j] - means[k]
            comps.append(
                {
                    "j": j,
                    "k": k,
                    "diff": diff,
                    "lower": diff - q * se,
                    "upper": diff + q * se,
                    "p_adj": 1.0 - _ptukey(abs(diff) / se, 1.0, float(J), float(df)),
                }
            )
    return RichResult(
        title="Tukey-Kramer intervals",
        summary_lines=[("q", q), ("MSWG", mswg), ("df", df)],
        payload={"q": q, "mswg": mswg, "df": df, "comparisons": comps},
    )


def cheatsheet():
    return "tukeykramer: Tukey-Kramer all-pairs intervals with the studentized range. Wilcox (2017) eq (12.4)."
