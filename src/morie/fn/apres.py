# morie.fn -- function file from book-equation translation pipeline (rootcoder007/morie)
"""Aggregate PRE across roll calls."""

from __future__ import annotations

from ._containers import DescriptiveResult


def apre_statistic(all_pre, minority=None) -> DescriptiveResult:
    """Aggregate proportional reduction in error across roll calls.

    Poole and Rosenthal's APRE pools the classification errors over roll
    calls before dividing,

        APRE = sum_j (m_j - e_j) / sum_j m_j,

    with ``m_j`` the size of the minority side of roll call ``j`` and
    ``e_j`` the number of votes the spatial model misclassifies. Since the
    per-vote PRE is ``(m_j - e_j) / m_j``, APRE is the minority-weighted
    mean of the PREs: lopsided votes, which carry little information, get
    little weight. Pass the minority sizes; without them every roll call
    is weighted equally and the result is the plain mean of the PREs
    (``weighted`` is then False).

    Parameters
    ----------
    all_pre : array-like
        PRE of each roll call.
    minority : array-like, optional
        Minority-side size of each roll call (positive).

    Returns
    -------
    DescriptiveResult
        ``value`` = APRE; ``extra`` has ``mean_pre``, ``median_pre``,
        ``min_pre``, ``max_pre``, ``n_roll_calls``, ``weighted``.

    References
    ----------
    Poole, K. T. and Rosenthal, H. (1997). Congress: A Political-Economic
    History of Roll Call Voting. Oxford University Press, ch. 2.
    Armstrong, D. A. et al. (2021). Analyzing Spatial Models of Choice and
    Judgment (2nd ed.). CRC Press, ch. 5.

    Examples
    --------
    >>> round(apre_statistic([0.8, 0.5], minority=[40, 10]).value, 12)
    0.74
    """
    pre = [float(v) for v in (all_pre.tolist() if hasattr(all_pre, "tolist") else all_pre)]
    k = len(pre)
    if k == 0:
        raise ValueError("all_pre is empty")
    if minority is None:
        w = [1.0] * k
    else:
        w = [float(v) for v in (minority.tolist() if hasattr(minority, "tolist") else minority)]
        if len(w) != k:
            raise ValueError("minority must have one entry per roll call")
        if any(not v > 0 for v in w):
            raise ValueError("minority sizes must be positive")
    num = 0.0
    den = 0.0
    tot = 0.0
    for p, m in zip(pre, w):
        num += m * p
        den += m
        tot += p
    apre = num / den
    s = sorted(pre)
    med = s[k // 2] if k % 2 else 0.5 * (s[k // 2 - 1] + s[k // 2])
    return DescriptiveResult(
        name="apre_statistic",
        value=apre,
        extra={
            "mean_pre": tot / k,
            "median_pre": med,
            "min_pre": s[0],
            "max_pre": s[-1],
            "n_roll_calls": k,
            "weighted": minority is not None,
        },
    )


apres = apre_statistic


def cheatsheet() -> str:
    return "apre_statistic(all_pre, minority) -> APRE = sum(m_j PRE_j) / sum(m_j)."


# compact alias per ledger/NAMING.md
aprestatistic = apre_statistic
