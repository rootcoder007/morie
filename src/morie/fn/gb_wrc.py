# morie.fn -- function file (rootcoder007/morie)
"""Critical regions for the runs test from the exact null distribution."""

import math

from ._richresult import RichResult

__all__ = ['runscrit', 'gibbons_runs_critical']


def runscrit(n1, n2, alpha=0.05, tail="two-sided"):
    """Largest exact-level rejection region for the total-runs test.

    Section 3.2 (book p. 84).  Table D is entered to find the largest
    left-tail critical value with P(R <= c) <= alpha (clustering
    alternative), the smallest right-tail value with P(R >= c) <= alpha
    (mixing alternative), or both at alpha/2.  The realised sizes are
    returned because the distribution is discrete and the nominal
    level is essentially never attained.

    Parameters
    ----------
    n1, n2 : int
        Counts of the two element types.
    alpha : float, optional
        Nominal level (default 0.05).
    tail : str, optional
        ``"two-sided"``, ``"left"`` or ``"right"``.

    Returns
    -------
    RichResult
        keys ``lower``, ``upper``, ``alpha_lower``, ``alpha_upper``,
        ``alpha_exact``, ``n1``, ``n2``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 3.2, p. 84; Table D.
    """
    n1 = int(n1)
    n2 = int(n2)
    alpha = float(alpha)
    if n1 < 1 or n2 < 1:
        raise ValueError("n1 and n2 must be at least 1.")
    if tail not in ("two-sided", "left", "right"):
        raise ValueError("tail must be two-sided, left or right.")
    n = n1 + n2
    den = math.comb(n, n1)
    support = list(range(2, n + 1))
    pmf = []
    for rr in support:
        if rr % 2 == 0:
            k = rr // 2
            p = 2.0 * math.comb(n1 - 1, k - 1) * math.comb(n2 - 1, k - 1)
        else:
            k = (rr - 1) // 2
            p = (
                math.comb(n1 - 1, k - 1) * math.comb(n2 - 1, k)
                + math.comb(n1 - 1, k) * math.comb(n2 - 1, k - 1)
            )
        pmf.append(p / den)
    a = alpha / 2.0 if tail == "two-sided" else alpha
    lower = float("nan")
    al = 0.0
    if tail in ("two-sided", "left"):
        acc = 0.0
        for i, s in enumerate(support):
            acc += pmf[i]
            if acc <= a:
                lower = float(s)
                al = acc
            else:
                break
    upper = float("nan")
    au = 0.0
    if tail in ("two-sided", "right"):
        acc = 0.0
        for i in range(len(support) - 1, -1, -1):
            acc += pmf[i]
            if acc <= a:
                upper = float(support[i])
                au = acc
            else:
                break
    return RichResult(
        payload={
            "lower": lower,
            "upper": upper,
            "alpha_lower": float(al),
            "alpha_upper": float(au),
            "alpha_exact": float(al + au),
            "n1": n1,
            "n2": n2,
            "method": "exact runs-test critical region (Table D, Sec. 3.2)",
        }
    )


gibbons_runs_critical = runscrit
