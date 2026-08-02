# morie.fn -- function file (rootcoder007/morie)
"""Party unity score per legislator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["party_unity_score"]


def party_unity_score(vote_matrix, party_id, unity_votes_only=False):
    r"""How often each legislator votes with their party majority.

    .. math:: \mathrm{Unity}_i = \frac{\#\{j : v_{ij} =
              \text{majority}_{p(i), j}\}}{\#\{j : v_{ij}
              \text{ cast, majority defined}\}}.

    With ``unity_votes_only=True`` only "party unity votes" count --
    roll calls where the party majorities oppose each other, the CQ
    convention -- otherwise every roll call with a defined party
    majority counts. Ties within a party leave that roll call
    undefined for its members.

    Parameters
    ----------
    vote_matrix : array-like, shape (n, q)
        Binary votes (1 = yea, 0 = nay, NaN = missing).
    party_id : array-like, shape (n,)
        Party label per legislator.
    unity_votes_only : bool, default False
        Restrict to roll calls where the two largest parties'
        majorities oppose each other.

    Returns
    -------
    RichResult
        keys: ``unity`` (n,), ``by_party`` (dict party -> mean),
        ``n_votes_scored`` (n,), ``n``, ``method``.

    References
    ----------
    Poole, K. T. & Rosenthal, H. (1997). *Congress*. Oxford
    University Press. (party voting and unity scores)

    Rice, S. A. (1925). The behavior of legislative groups.
    *Political Science Quarterly*, 40(1), 60-72. (the cohesion
    tradition the score belongs to)
    """
    V = np.asarray(vote_matrix, dtype=float)
    if V.ndim != 2:
        raise ValueError("vote_matrix must be 2-D.")
    pid = np.asarray(party_id).ravel()
    n, q = V.shape
    if pid.size != n:
        raise ValueError("party_id must have one entry per legislator.")

    parties = list(dict.fromkeys(pid.tolist()))
    # party majority position per roll call; NaN when tied or empty
    maj = {}
    for p in parties:
        rows = V[pid == p]
        m = np.full(q, np.nan)
        for j in range(q):
            col = rows[:, j]
            col = col[~np.isnan(col)]
            if col.size == 0:
                continue
            yea = col.sum()
            nay = col.size - yea
            if yea > nay:
                m[j] = 1.0
            elif nay > yea:
                m[j] = 0.0
        maj[p] = m

    scored = np.ones(q, dtype=bool)
    if unity_votes_only:
        if len(parties) < 2:
            raise ValueError("unity_votes_only needs at least two parties.")
        sizes = [(p, int((pid == p).sum())) for p in parties]
        big2 = [p for p, _ in sorted(sizes, key=lambda t: -t[1])[:2]]
        a, b = maj[big2[0]], maj[big2[1]]
        scored = ~np.isnan(a) & ~np.isnan(b) & (a != b)

    unity = np.full(n, np.nan)
    counts = np.zeros(n, dtype=int)
    for i in range(n):
        m = maj[pid[i]]
        ok = scored & ~np.isnan(V[i]) & ~np.isnan(m)
        if ok.sum() > 0:
            unity[i] = float((V[i, ok] == m[ok]).mean())
            counts[i] = int(ok.sum())

    by_party = {
        p: float(np.nanmean(unity[pid == p])) if np.any(~np.isnan(unity[pid == p])) else float("nan")
        for p in parties
    }
    return RichResult(
        payload={
            "unity": unity,
            "by_party": by_party,
            "n_votes_scored": counts,
            "n": int(n),
            "method": "Party unity score (share of votes cast with the party majority)",
        }
    )


def cheatsheet():
    return "agpar: Unity_i = votes with own party majority / votes cast (CQ variant optional)"
