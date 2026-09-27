"""Majority tournaments over spatial alternatives: Condorcet sets, top cycle, uncovered set, Banks set.

Miller, N. R. (1980). A new solution set for tournaments and majority voting. American Journal of
Political Science 24, 68-96 (uncovered set). Banks, J. S. (1985). Sophisticated voting outcomes
and agenda control. Social Choice and Welfare 1, 295-306 (Banks set). Schwartz, T. (1972).
Rationality and the myth of the maximum. Nous 6, 97-117 (top cycle). Copeland, A. H. (1951). A
reasonable social welfare function. University of Michigan seminar notes. Borda, J.-C. de (1781).
Memoire sur les elections au scrutin.
"""

from ._richresult import RichResult
from .svutil import voter_utility

__all__ = ["majority_tournament"]


def _reach(beats, m):
    R = [[beats[i][j] for j in range(m)] for i in range(m)]
    for k in range(m):
        for i in range(m):
            if R[i][k]:
                for j in range(m):
                    if R[k][j]:
                        R[i][j] = True
    return R


def majority_tournament(voters, alternatives, model="quadratic", **utility_args):
    r"""Pairwise majority relation among alternatives and the tournament solutions.

    Voter i prefers a to b when U_i(a) > U_i(b) (ties abstain); a beats b when more voters
    prefer a. From this relation: the Condorcet winner (beats all) and loser (loses to all);
    ``cyclic`` (the relation is not acyclic); the top cycle (Smith set: the smallest set whose
    members beat everyone outside, i.e. the alternatives that reach every other through a
    chain of majority wins); the uncovered set (a covers b when a beats b and a beats every
    alternative b beats; Miller 1980); the Banks set (maxima of maximal transitive chains,
    found by exhaustive search, Banks 1985); Copeland scores (wins - losses); and Borda scores
    (sum over voters of the number of alternatives ranked below).

    Parameters
    ----------
    voters, alternatives : points
    model : str
        Utility model (see :func:`voter_utility`).
    **utility_args
        Passed to :func:`voter_utility`.

    Returns
    -------
    RichResult
        Keys: margin (a x a vote margins), beats, condorcet_winner, condorcet_loser, cyclic,
        top_cycle, uncovered, banks, copeland, borda.

    References
    ----------
    Miller, N. R. (1980). American Journal of Political Science 24, 68-96.
    Banks, J. S. (1985). Social Choice and Welfare 1, 295-306.

    Examples
    --------
    >>> majority_tournament([[0.0], [1.0], [2.0]], [[0.0], [1.0], [2.0]])["condorcet_winner"]
    1
    """
    U = voter_utility(voters, alternatives, model=model, **utility_args)["utility"]
    m = len(U[0])
    M = [[0] * m for _ in range(m)]
    for row in U:
        for a in range(m):
            for b in range(m):
                if row[a] > row[b]:
                    M[a][b] += 1
    margin = [[M[a][b] - M[b][a] for b in range(m)] for a in range(m)]
    beats = [[margin[a][b] > 0 for b in range(m)] for a in range(m)]
    winner = next((a for a in range(m) if all(beats[a][b] for b in range(m) if b != a)), None)
    loser = next((a for a in range(m) if all(beats[b][a] for b in range(m) if b != a)), None)
    R = _reach(beats, m)
    cyclic = any(R[a][a] for a in range(m))
    # Smith set: alternatives that reach (weakly, via beats-or-ties chains) every other alternative
    weak = [[margin[a][b] >= 0 and a != b for b in range(m)] for a in range(m)]
    W = _reach(weak, m)
    top = [a for a in range(m) if all(W[a][b] for b in range(m) if b != a)]
    uncovered = [
        b
        for b in range(m)
        if not any(beats[a][b] and all(beats[a][c] for c in range(m) if beats[b][c]) for a in range(m) if a != b)
    ]
    banks = set()

    def extend(chain):
        # a transitive chain grows only upward: a newcomer must beat every member;
        # when none can, the chain is maximal and its top is a Banks point
        grew = False
        for c in range(m):
            if c not in chain and all(beats[c][x] for x in chain):
                grew = True
                extend(chain + [c])
        if not grew:
            banks.add(chain[-1])

    for a in range(m):
        extend([a])
    copeland = [sum(1 for b in range(m) if beats[a][b]) - sum(1 for b in range(m) if beats[b][a]) for a in range(m)]
    borda = [sum(sum(1 for b in range(m) if row[a] > row[b]) for row in U) for a in range(m)]
    return RichResult(
        title="Majority tournament",
        summary_lines=[("Condorcet winner", winner)],
        payload={
            "margin": margin,
            "beats": beats,
            "condorcet_winner": winner,
            "condorcet_loser": loser,
            "cyclic": cyclic,
            "top_cycle": top,
            "uncovered": uncovered,
            "banks": sorted(banks),
            "copeland": copeland,
            "borda": borda,
        },
    )


def cheatsheet():
    return "svtour: majority tournament solutions (Condorcet, top cycle, uncovered, Banks, Copeland, Borda)"
