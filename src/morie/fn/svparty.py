"""Party positions from member ideal points or expert placements, with bootstrap standard errors.

Laver, M. and Hunt, W. B. (1992). Policy and Party Competition. Routledge (expert surveys,
positions as mean placements). Poole, K. T. and Rosenthal, H. (1997). Congress: A
Political-Economic History of Roll Call Voting. Oxford University Press (party medians).
Efron, B. and Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman and Hall.
"""

from ._metaheur import Rand
from ._richresult import RichResult

__all__ = ["party_positions"]


def _med(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def party_positions(points, party, statistic="mean", n_boot=200, seed=0):
    r"""Position of each party on each dimension as the mean (expert placements, Laver and Hunt
    1992) or median (legislator ideal points, the congressional party median) of its members,
    with nonparametric bootstrap standard errors (members resampled within party with Philox
    draws) and the distance between each pair of parties.

    Parameters
    ----------
    points : list of numbers or points
    party : sequence of labels
    statistic : {"mean", "median"}
    n_boot : int
    seed : int

    Returns
    -------
    RichResult
        Keys: parties (sorted labels), position, se, n, distance (party x party).

    References
    ----------
    Laver, M. and Hunt, W. B. (1992). Policy and Party Competition.
    Efron, B. and Tibshirani, R. J. (1993). An Introduction to the Bootstrap.

    Examples
    --------
    >>> party_positions([1, 2, 3, 7, 8, 9], ["D", "D", "D", "R", "R", "R"], n_boot=0)["position"]
    [[2.0], [8.0]]
    """
    P = [[float(t)] for t in points] if isinstance(points[0], (int, float)) else [[float(t) for t in r] for r in points]
    d = len(P[0])
    labs = sorted(set(party), key=str)
    stat = (lambda v: sum(v) / len(v)) if statistic == "mean" else _med
    if statistic not in ("mean", "median"):
        raise ValueError('statistic must be "mean" or "median"')
    rnd = Rand(seed)
    pos, se, cnt = [], [], []
    for lab in labs:
        mem = [P[i] for i in range(len(P)) if party[i] == lab]
        cnt.append(len(mem))
        pos.append([stat([m[j] for m in mem]) for j in range(d)])
        if n_boot:
            reps = []
            for _ in range(int(n_boot)):
                idx = [rnd.idx(len(mem)) for _ in range(len(mem))]
                reps.append([stat([mem[i][j] for i in idx]) for j in range(d)])
            sej = []
            for j in range(d):
                m = sum(r[j] for r in reps) / len(reps)
                sej.append((sum((r[j] - m) ** 2 for r in reps) / (len(reps) - 1)) ** 0.5)
            se.append(sej)
    dist = [
        [sum((pos[a][j] - pos[b][j]) ** 2 for j in range(d)) ** 0.5 for b in range(len(labs))] for a in range(len(labs))
    ]
    return RichResult(
        title="Party positions",
        summary_lines=[("parties", len(labs))],
        payload={"parties": labs, "position": pos, "se": se if n_boot else None, "n": cnt, "distance": dist},
    )


def cheatsheet():
    return "svparty: party positions (mean or median) with bootstrap standard errors"
