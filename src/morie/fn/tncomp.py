# morie.fn -- function file (rootcoder007/morie)
r"""Dissimilarity-based compound selection: MaxMin and MaxSum.

**The task.** Pick :math:`k` compounds from :math:`N` so that the
selection covers the collection rather than clumping in whatever
region happens to be over-represented. Both algorithms here are
greedy and differ only in the objective used at each step, with
:math:`d = 1 - T` the Tanimoto distance:

.. math:: \text{MaxMin: } \arg\max_{i \notin S}
          \min_{j \in S} d(i, j), \qquad
          \text{MaxSum: } \arg\max_{i \notin S}
          \sum_{j \in S} d(i, j).

**Why the difference matters.** MaxSum maximises a total, so a
compound far from most of the selection wins even if it sits almost
on top of one member -- the large distances outvote the small one.
MaxMin maximises the *worst* distance, so nothing is added next to
something already chosen. Snarey et al. found MaxMin gives better
coverage of a collection, and the anchor exhibits the mechanism
directly: on a set with four well-separated groups and a crowd of
near-duplicates in one of them, MaxMin takes one from each group and
MaxSum does not.

Neither is optimal -- maximum diversity is NP-hard -- and neither is
random: both are deterministic given the seed, and the seed is
reported.

References
----------
Snarey, M., Terrett, N. K., Willett, P. & Wilton, D. J. (1997)
"Comparison of algorithms for dissimilarity-based compound
selection", *Journal of Molecular Graphics and Modelling* 15(6),
372-385, doi:10.1016/S1093-3263(98)00008-4. The MaxMin and MaxSum
objectives reproduced above, their greedy implementation, and the
finding that MaxMin covers a collection better.

Willett, P., Barnard, J. M. & Downs, G. M. (1998) "Chemical
similarity searching", *Journal of Chemical Information and Computer
Sciences* 38(6), 983-996, doi:10.1021/ci9800211, for the Tanimoto
distance the objectives are measured in; see :mod:`morie.fn.sasimi`.
"""

from ._richresult import RichResult
from .sasimi import fingerprint, tanimoto

__all__ = ["distance_matrix", "maxmin_selection", "maxsum_selection",
           "diversity", "OBJECTIVES", "maxmin_diversity"]

OBJECTIVES = ("maxmin", "maxsum")


def distance_matrix(fps):
    r"""All pairwise Tanimoto distances."""
    F = [fingerprint(x) for x in fps]
    if len(F) < 2:
        raise ValueError("tncomp: need at least two compounds")
    n = len(F)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            D[i][j] = D[j][i] = 1.0 - tanimoto(F[i], F[j])
    return D


def _seed(D, seed):
    if seed is not None:
        s = int(seed)
        if not 0 <= s < len(D):
            raise ValueError("tncomp: seed %d is not a compound "
                             "index" % s)
        return s
    # Deterministic default: the compound furthest from the rest.
    tot = [sum(row) for row in D]
    return max(range(len(D)), key=lambda i: (tot[i], -i))


def _select(fps, k, objective, seed=None, D=None):
    if objective not in OBJECTIVES:
        raise ValueError("tncomp: objective must be one of %s, got "
                         "%r" % (", ".join(OBJECTIVES), objective))
    M = distance_matrix(fps) if D is None else D
    n = len(M)
    kk = int(k)
    if not 1 <= kk <= n:
        raise ValueError("tncomp: k must lie in [1, %d], got %d"
                         % (n, kk))
    chosen = [_seed(M, seed)]
    while len(chosen) < kk:
        rest = [i for i in range(n) if i not in chosen]
        if objective == "maxmin":
            score = {i: min(M[i][j] for j in chosen) for i in rest}
        else:
            score = {i: sum(M[i][j] for j in chosen) for i in rest}
        chosen.append(max(rest, key=lambda i: (score[i], -i)))
    return chosen, M


def maxmin_selection(fps, k, seed=None):
    r"""Greedy MaxMin: maximise the worst distance to the selection."""
    return _select(fps, k, "maxmin", seed)[0]


def maxsum_selection(fps, k, seed=None):
    r"""Greedy MaxSum: maximise the total distance to the selection."""
    return _select(fps, k, "maxsum", seed)[0]


def diversity(fps, subset, D=None):
    r"""How spread out a selection is.

    ``min_distance`` is the quantity MaxMin optimises and the one that
    says whether anything was picked twice over.
    """
    M = distance_matrix(fps) if D is None else D
    S = list(subset)
    if len(S) < 2:
        raise ValueError("tncomp: diversity needs at least two "
                         "selected compounds")
    if len(set(S)) != len(S):
        raise ValueError("tncomp: the selection repeats a compound")
    ds = [M[a][b] for i, a in enumerate(S) for b in S[i + 1:]]
    return {"min_distance": min(ds), "mean_distance":
            sum(ds) / float(len(ds)), "max_distance": max(ds),
            "n_pairs": len(ds)}


def maxmin_diversity(fps, k, objective="maxmin", seed=None):
    r"""Entry point: select ``k`` diverse compounds."""
    chosen, M = _select(fps, k, objective, seed)
    out = {"estimate": chosen, "selection": chosen,
           "objective": objective, "k": int(k),
           "seed": chosen[0], "n_compounds": len(M),
           "method": "Snarey et al. (1997) greedy %s selection on "
                     "Tanimoto distance" % objective}
    if len(chosen) > 1:
        out.update(diversity(fps, chosen, M))
    return RichResult(payload=out)
