"""Amendment agendas under sincere and sophisticated voting.

Farquharson, R. (1969). Theory of Voting. Yale University Press. McKelvey, R. D. and Niemi, R. G.
(1978). A multistage game representation of sophisticated voting for binary procedures. Journal
of Economic Theory 18, 1-22. Shepsle, K. A. and Weingast, B. R. (1984). Uncovered sets and
sophisticated voting outcomes with implications for agenda institutions. American Journal of
Political Science 28, 49-74.
"""

from ._richresult import RichResult
from .svutil import voter_utility

__all__ = ["amendment_agenda"]


def amendment_agenda(voters, agenda, model="quadratic", **utility_args):
    r"""Outcome of the amendment procedure over ``agenda`` = [a_1, ..., a_m] (a_m typically the status quo).

    a_1 meets a_2, the survivor meets a_3, and so on. Sincere voting compares the two
    alternatives on the floor; sophisticated voting compares the final outcomes each branch
    leads to, found by backward induction on the voting tree (McKelvey and Niemi 1978). The
    sophisticated outcome always lies in the uncovered set (Shepsle and Weingast 1984).

    Parameters
    ----------
    voters : points
    agenda : points, in voting order
    model : str
    **utility_args
        Passed to :func:`voter_utility`.

    Returns
    -------
    RichResult
        Keys: sincere, sophisticated (0-based agenda indices), sincere_path.

    References
    ----------
    McKelvey, R. D. and Niemi, R. G. (1978). Journal of Economic Theory 18, 1-22.
    Shepsle, K. A. and Weingast, B. R. (1984). American Journal of Political Science 28, 49-74.

    Examples
    --------
    >>> amendment_agenda([[0.0], [0.4], [1.0]], [[1.0], [0.0], [0.5]])["sincere"]
    2
    """
    U = voter_utility(voters, agenda, model=model, **utility_args)["utility"]
    m = len(U[0])

    def maj(a, b):
        pa = sum(1 for row in U if row[a] > row[b])
        pb = sum(1 for row in U if row[b] > row[a])
        return a if pa >= pb else b  # ties keep the earlier (incumbent) alternative

    path = [0]
    w = 0
    for k in range(1, m):
        w = maj(w, k)
        path.append(w)

    memo = {}

    def soph(cur, k):
        if k == m:
            return cur
        key = (cur, k)
        if key not in memo:
            keep, take = soph(cur, k + 1), soph(k, k + 1)
            memo[key] = keep if maj(keep, take) == keep else take
        return memo[key]

    return RichResult(
        title="Amendment agenda",
        summary_lines=[("sincere", path[-1])],
        payload={"sincere": path[-1], "sophisticated": soph(0, 1), "sincere_path": path},
    )


def cheatsheet():
    return "svagnd: amendment agenda outcomes, sincere and sophisticated (backward induction)"
