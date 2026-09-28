# morie.fn -- function file (rootcoder007/morie)
"""Electoral systems and legislative behaviour: the Gelman-King incumbency advantage, proportional seat
allocation (highest averages and largest remainders), the two-round runoff, and network (modularity)
polarization of roll-call agreement."""

from __future__ import annotations

import math

from ._qpcore import inverse, ssum
from ._richresult import RichResult
from .communities import graph_modularity

__all__ = ["gelman_king_incumbency", "proportional_seats", "runoff_winner", "network_polarization"]


def gelman_king_incumbency(v, v_lag, party, incumbency) -> RichResult:
    r"""Gelman-King estimator of the incumbency advantage in two-party vote shares.

    OLS of ``v_t = b0 + b1 v_{t-1} + b2 P_t + psi I_t + e`` over districts,
    with ``P`` the party of the previous winner (+1 Democrat, -1 Republican)
    and ``I`` incumbency (+1 Democratic incumbent running, -1 Republican,
    0 open seat); ``psi`` is the incumbency advantage. Returns all
    coefficients, their OLS standard errors and ``psi``.

    References
    ----------
    Gelman, A. and King, G. (1990). Estimating incumbency advantage without
    bias. *American Journal of Political Science*, 34, 1142-1164.

    Examples
    --------
    >>> r = gelman_king_incumbency([0.55, 0.62, 0.41, 0.48, 0.66, 0.37], [0.52, 0.58, 0.45, 0.50, 0.61, 0.40],
    ...                            [1, 1, -1, -1, 1, -1], [1, 1, -1, 0, 1, -1])
    >>> round(r.psi, 10)
    0.0060550459
    """
    n = len(v)
    X = [[1.0, float(v_lag[i]), float(party[i]), float(incumbency[i])] for i in range(n)]
    y = [float(t) for t in v]
    p = 4
    XtX = [[ssum(X[i][a] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    Xty = [ssum(X[i][a] * y[i] for i in range(n)) for a in range(p)]
    Ai = inverse(XtX)
    b = [ssum(Ai[a][c] * Xty[c] for c in range(p)) for a in range(p)]
    res = [y[i] - ssum(X[i][a] * b[a] for a in range(p)) for i in range(n)]
    s2 = ssum(r * r for r in res) / (n - p)
    se = [math.sqrt(s2 * Ai[a][a]) for a in range(p)]
    return RichResult(payload={"coefficients": b, "se": se, "psi": b[3], "sigma": math.sqrt(s2)})


def proportional_seats(votes, seats: int, method: str = "dhondt", *, threshold: float = 0.0) -> RichResult:
    r"""Allocate ``seats`` proportionally to ``votes``.

    Highest-averages methods give each successive seat to the largest
    ``votes_i / d(s_i)`` with divisors ``d(s) = s + 1`` (``"dhondt"``,
    Jefferson), ``2s + 1`` (``"sainte_lague"``, Webster) or ``1.4`` then
    ``2s + 1`` (``"modified_sainte_lague"``); ties go to the party with more
    votes, then the lower index. Largest-remainder methods (``"hare"`` with
    quota ``V/S``, ``"droop"`` with ``floor(V/(S + 1)) + 1``) give
    ``floor(v/q)`` seats and the rest by largest remainders. Parties below
    ``threshold`` (a fraction of the vote) receive nothing.

    References
    ----------
    Balinski, M. L. and Young, H. P. (2001). *Fair Representation*, 2nd edn.
    Brookings Institution Press.
    Gallagher, M. (1991). Proportionality, disproportionality and electoral
    systems. *Electoral Studies*, 10, 33-51.

    Examples
    --------
    >>> proportional_seats([100000, 80000, 30000, 20000], 8).seats
    [4, 3, 1, 0]
    >>> proportional_seats([100000, 80000, 30000, 20000], 8, "sainte_lague").seats
    [3, 3, 1, 1]
    """
    V = [float(v) for v in votes]
    tot = ssum(V)
    ok = [v / tot >= threshold for v in V]
    k = len(V)
    s = [0] * k
    if method in ("dhondt", "sainte_lague", "modified_sainte_lague"):

        def div(n):
            if method == "dhondt":
                return n + 1.0
            if method == "modified_sainte_lague" and n == 0:
                return 1.4
            return 2.0 * n + 1.0

        for _ in range(seats):
            j = max((i for i in range(k) if ok[i]), key=lambda i: (V[i] / div(s[i]), V[i], -i))
            s[j] += 1
    elif method in ("hare", "droop"):
        tv = ssum(V[i] for i in range(k) if ok[i])
        q = tv / seats if method == "hare" else math.floor(tv / (seats + 1)) + 1
        for i in range(k):
            if ok[i]:
                s[i] = int(math.floor(V[i] / q))
        rem = sorted((i for i in range(k) if ok[i]), key=lambda i: (-(V[i] / q - s[i]), -V[i], i))
        left = seats - sum(s)
        for i in rem[: max(left, 0)]:
            s[i] += 1
    else:
        raise ValueError("unknown method")
    return RichResult(payload={"seats": s, "method": method})


def runoff_winner(ballots, n_candidates: int | None = None) -> RichResult:
    r"""Two-round (contingent) runoff from ranked ballots.

    A candidate with more than half of the first preferences wins outright;
    otherwise the two candidates with the most first preferences (ties to
    the lower index) meet in a second round in which each ballot supports
    whichever of the two it ranks higher (ballots ranking neither abstain).
    Candidates are integers ``0 .. m-1``; ties in the runoff go to the
    candidate with more first preferences.

    References
    ----------
    Blais, A., Massicotte, L. and Dobrzynska, A. (1997). Direct presidential
    elections: a world summary. *Electoral Studies*, 16, 441-455.

    Examples
    --------
    >>> runoff_winner([[0, 1, 2]] * 4 + [[1, 2, 0]] * 3 + [[2, 1, 0]] * 2).winner
    1
    """
    B = [[int(c) for c in b] for b in ballots]
    m = n_candidates if n_candidates is not None else 1 + max(c for b in B for c in b)
    first = [0] * m
    for b in B:
        if b:
            first[b[0]] += 1
    nv = ssum(first)
    top = max(range(m), key=lambda c: (first[c], -c))
    if first[top] > nv / 2:
        return RichResult(payload={"winner": top, "first_round": first, "second_round": None})
    order = sorted(range(m), key=lambda c: (-first[c], c))
    a, b2 = order[0], order[1]
    va = vb = 0
    for b in B:
        ra = b.index(a) if a in b else math.inf
        rb = b.index(b2) if b2 in b else math.inf
        if ra < rb:
            va += 1
        elif rb < ra:
            vb += 1
    win = a if (va, first[a], -a) > (vb, first[b2], -b2) else b2
    return RichResult(payload={"winner": win, "first_round": first, "second_round": {a: va, b2: vb}})


def network_polarization(votes, party) -> RichResult:
    r"""Party polarization as the modularity of the roll-call agreement network (Waugh et al. 2009).

    ``votes`` is a legislators x roll calls matrix of 1 (yea), 0 (nay) or
    ``None`` (absent); edge weights are the proportion of shared roll calls
    on which two legislators voted alike. The modularity of the partition by
    ``party`` measures how much more agreement lies within than between
    parties (0: none; larger: more polarized).

    References
    ----------
    Waugh, A. S., Pei, L., Fowler, J. H., Mucha, P. J. and Porter, M. A.
    (2009). Party polarization in Congress: a network science approach.
    arXiv:0907.3509.

    Examples
    --------
    >>> V = [[1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1]]
    >>> round(network_polarization(V, ["D", "D", "R", "R"]).modularity, 10)
    0.2040816327
    """
    n = len(votes)
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            same = tot = 0
            for a, b in zip(votes[i], votes[j]):
                if (
                    a is None
                    or b is None
                    or (isinstance(a, float) and math.isnan(a))
                    or (isinstance(b, float) and math.isnan(b))
                ):
                    continue
                tot += 1
                same += 1 if a == b else 0
            A[i][j] = A[j][i] = same / tot if tot else 0.0
    return RichResult(payload={"modularity": graph_modularity(A, list(party)), "agreement": A})


def cheatsheet() -> str:
    return (
        "gelman_king_incumbency / proportional_seats / runoff_winner / network_polarization -> electoral systems "
        "and legislative polarization."
    )
