# morie.fn -- function file (rootcoder007/morie)
"""Cohen's kappa for two raters.

Source: Cohen, J. (1960), "A coefficient of agreement for nominal
scales", *Educational and Psychological Measurement* 20(1):37-46.  The
1960 article is paywalled and was NOT read directly; the estimator was
checked against the standard published statement of it:

    kappa = (p_o - p_e) / (1 - p_e)
    p_o   = (1/N) sum_k O_kk
    p_e   = (1/N^2) sum_k O_k. O_.k

i.e. p_e is formed from the product of the row and column marginals,
treating the two raters as independent.

The standard error reported here is the large-sample approximation
se = sqrt(p_o (1 - p_o) / N) / (1 - p_e).  The exact variance of
Fleiss, Cohen & Everitt (1969) could NOT be obtained, so it is not
implemented; treat ``se`` as the conventional approximation, not as
that paper's expression.
"""

import math

from ._richresult import RichResult

__all__ = ["cohens_kappa"]


def _levels(a, b):
    seen = []
    for v in list(a) + list(b):
        if v not in seen:
            seen.append(v)
    seen.sort()
    return seen


def cohens_kappa(rater1, rater2):
    """Cohen's kappa for two raters on a nominal scale.

    Parameters
    ----------
    rater1, rater2 : sequence
        Categorical ratings of the same n subjects.

    Returns
    -------
    RichResult
        ``kappa``, ``p_observed``, ``p_expected``, ``se``, ``z``,
        ``n``, ``n_categories``.
    """
    r1 = list(rater1)
    r2 = list(rater2)
    n = len(r1)
    if n == 0 or len(r2) != n:
        raise ValueError("rater1 and rater2 must be non-empty and of equal length")
    lv = _levels(r1, r2)
    k = len(lv)
    pos = {v: i for i, v in enumerate(lv)}
    tab = [[0.0] * k for _ in range(k)]
    for a, b in zip(r1, r2):
        tab[pos[a]][pos[b]] += 1.0
    row = [sum(tab[i]) for i in range(k)]
    col = [sum(tab[i][j] for i in range(k)) for j in range(k)]
    po = sum(tab[i][i] for i in range(k)) / n
    pe = sum(row[i] * col[i] for i in range(k)) / (n * n)
    kap = (po - pe) / (1.0 - pe) if pe < 1.0 else 0.0
    se = (math.sqrt(po * (1.0 - po) / n) / (1.0 - pe)) if pe < 1.0 else 0.0
    z = kap / se if se > 0.0 else 0.0
    return RichResult(payload={
        "kappa": float(kap), "p_observed": float(po),
        "p_expected": float(pe), "se": float(se), "z": float(z),
        "n": n, "n_categories": k,
        "method": "Cohen (1960) kappa, two raters, nominal scale"})


def cheatsheet():
    return "kappac: Cohen (1960) kappa for two raters"


# compact alias per ledger/NAMING.md
kappaco = cohens_kappa
