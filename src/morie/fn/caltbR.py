# morie.fn -- function file (rootcoder007/morie)
r"""Calibrated recommendations: keep the user's proportions.

A user who watched 70 romance and 30 action films should get roughly
70/30 back. Ranking by accuracy does not deliver that, and the reason
is structural rather than a modelling defect.

**Accuracy actively rewards crowding out the minority interest.** With
only genre information, the situation is the imbalanced classification
problem: predicting the majority label for everything is optimal. In
the running example the top ten titles by :math:`p(i \mid u)` are all
romance -- and even when each film has its own
:math:`p(i \mid \text{genre})`, the tenth-best romance still beats the
best action title, because
:math:`\frac{p(i_{g_r,10}|u)}{p(i_{g_a,1}|u)} \approx
\frac{1}{2.1}\cdot\frac{0.7}{0.3} > 1`
(the 2.1 measured on MovieLens 20M). So calibration must cost some
accuracy; that is a consequence of the arithmetic, not a tuning
failure.

**Two distributions, then a divergence between them.** The played
distribution :math:`p(g \mid u)` weights each film by
:math:`w_{u,i}` (eq. 2); the recommended distribution
:math:`q(g \mid u)` weights by rank, :math:`w_{r(i)}` (eq. 3), so an
MRR- or nDCG-style discount can be applied. Both accept fractional
genre membership :math:`p(g \mid i)`, so a film may belong to several
genres.

**KL, with one wrinkle.** :math:`C_{KL} = KL(p \| \tilde q)` with
:math:`\tilde q = (1-\alpha)q + \alpha p` and a small
:math:`\alpha` (0.01 in the paper), because :math:`q` can be zero
where :math:`p` is not. Three properties are wanted, and KL has all
three: it is zero exactly at :math:`p = \tilde q`; it is very
sensitive to small discrepancies where :math:`p` is small (2% played
but 1% recommended is a bigger sin than 50% played and 49%
recommended); and it prefers the less extreme error -- 31% of a genre
played 30% of the time scores better than 29%. That last property is
deliberate and asymmetric, and the anchor checks it against the
paper's printed table. The Hellinger distance is offered as the
milder alternative the paper names.

**Re-ranking, because calibration is a property of the list.**
Pointwise and pairwise training cannot see it, so it is applied as
post-processing by maximum marginal relevance,

.. math:: I^* = \arg\max_{|I| = N}\ (1-\lambda)\,s(I)
          - \lambda\, C_{KL}(p, q(I)),

built greedily. Greedy is not a concession here: the surrogate is
submodular, so each prefix of the list is :math:`(1-1/e)` optimal --
which matters because the user may only see the first few rows before
scrolling.

**Calibration is not diversity.** Diversity minimises redundancy; in a
two-genre world it would return 50/50, not the user's 70/30. It can
also introduce genres the user has never watched, which calibration
alone never will -- so a diversity prior is mixed in explicitly,
:math:`\bar p(g\mid u) = \beta p_0(g) + (1-\beta)p(g\mid u)`
(eq. 7), when escaping the filter bubble is wanted.

References
----------
Steck, H. (2018) "Calibrated Recommendations", *Proceedings of the
Twelfth ACM Conference on Recommender Systems (RecSys '18)*, 154-162,
doi:10.1145/3240323.3240372. Sec. 2.1 (the class-imbalance argument
and why accuracy favours the majority genre), Sec. 2.2 (the
varying-movie-probability example and the 2.1 figure from MovieLens
20M), Sec. 3 (eqs. 2-5: the two distributions, the rank weights, the
KL calibration metric, the alpha-smoothed q-tilde with alpha = 0.01,
the three desired properties, and the Hellinger alternative), Sec. 4
(eq. 6: maximum marginal relevance, greedy optimisation, submodularity
and the (1 - 1/e) guarantee for every prefix), Sec. 5.1 and Table 1
(calibration is not diversity; the printed comparison of C_KL,
BinomDiv and DP), and eq. 7 (the diversity prior).

Carbonell, J. & Goldstein, J. (1998) "The use of MMR, diversity-based
reranking for reordering documents and producing summaries",
*SIGIR '98*, 335-336, doi:10.1145/290941.291025. Maximum marginal
relevance.

Nemhauser, G. L., Wolsey, L. A. & Fisher, M. L. (1978) "An analysis of
approximations for maximizing submodular set functions - I",
*Mathematical Programming* 14, 265-294, doi:10.1007/BF01588971. The
(1 - 1/e) guarantee for greedy maximisation.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["genre_distribution", "calibration_kl",
           "calibration_hellinger", "diversity_prior",
           "calibrated_rerank"]

_EPS = 1e-12
_METRICS = ("kl", "hellinger")


def _norm(v):
    s = sum(v)
    if s <= _EPS:
        raise ValueError("caltbR: a genre distribution has no mass")
    return [x / s for x in v]


def genre_distribution(items, p_g_given_i, weights=None):
    r"""Eqs. (2)-(3): a weighted average of :math:`p(g \mid i)`.

    Used for both the played history (weights by recency, say) and the
    recommended list (weights by rank).
    """
    it = [int(v) for v in items]
    if not it:
        raise ValueError("caltbR: no items given")
    G = len(p_g_given_i[0])
    w = [1.0] * len(it) if weights is None \
        else [float(v) for v in k.vec(weights)]
    if len(w) != len(it):
        raise ValueError("caltbR: %d weights for %d items"
                         % (len(w), len(it)))
    tot = sum(w)
    if tot <= _EPS:
        raise ValueError("caltbR: the weights sum to zero")
    return [sum(w[n] * p_g_given_i[it[n]][g]
                for n in range(len(it))) / tot for g in range(G)]


def calibration_kl(p, q, alpha=0.01):
    r"""Eqs. (4)-(5): :math:`KL(p \,\|\, (1-\alpha)q + \alpha p)`."""
    pp = _norm([float(v) for v in k.vec(p)])
    qq = _norm([float(v) for v in k.vec(q)])
    if len(pp) != len(qq):
        raise ValueError("caltbR: %d genres in p but %d in q"
                         % (len(pp), len(qq)))
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError("caltbR: alpha must lie in (0,1), got %r"
                         % (alpha,))
    tot = 0.0
    for g in range(len(pp)):
        if pp[g] <= _EPS:
            continue
        qt = (1.0 - a) * qq[g] + a * pp[g]
        tot += pp[g] * math.log(pp[g] / max(qt, _EPS))
    return tot


def calibration_hellinger(p, q):
    r""":math:`\|\sqrt p - \sqrt q\|_2 / \sqrt 2`.

    Well defined at zeros, and less brutal than KL where :math:`p` is
    small -- the paper's named alternative.
    """
    pp = _norm([float(v) for v in k.vec(p)])
    qq = _norm([float(v) for v in k.vec(q)])
    s = sum((math.sqrt(pp[g]) - math.sqrt(qq[g])) ** 2
            for g in range(len(pp)))
    return math.sqrt(s) / math.sqrt(2.0)


def diversity_prior(p_u, p0, beta):
    r"""Eq. (7): :math:`\bar p = \beta p_0 + (1-\beta)p(g\mid u)`.

    Calibration alone never introduces an unplayed genre; this is what
    lets the list leave the filter bubble.
    """
    a = [float(v) for v in k.vec(p_u)]
    b = [float(v) for v in k.vec(p0)]
    t = float(beta)
    if not 0.0 <= t <= 1.0:
        raise ValueError("caltbR: beta must lie in [0,1], got %r"
                         % (beta,))
    if len(a) != len(b):
        raise ValueError("caltbR: prior has %d genres, target %d"
                         % (len(b), len(a)))
    return [t * b[g] + (1.0 - t) * a[g] for g in range(len(a))]


def calibrated_rerank(scores, p_g_given_i, p_target, N=10, lam=0.5,
                      metric="kl", alpha=0.01, rank_weights=None):
    r"""Eq. (6) by greedy maximisation of the submodular surrogate.

    Every prefix of the returned list is :math:`(1-1/e)` optimal, so a
    user who sees only the first few rows still gets a calibrated
    view.
    """
    if metric not in _METRICS:
        raise ValueError("caltbR: metric must be one of %s, got %r"
                         % (", ".join(_METRICS), metric))
    s = [float(v) for v in k.vec(scores)]
    n = len(s)
    if n == 0:
        raise ValueError("caltbR: no candidate items")
    Nn = min(int(N), n)
    if Nn < 1:
        raise ValueError("caltbR: N must be at least 1")
    lm = float(lam)
    if not 0.0 <= lm <= 1.0:
        raise ValueError("caltbR: lambda must lie in [0,1], got %r"
                         % (lam,))
    pt = _norm([float(v) for v in k.vec(p_target)])

    def cal(sel):
        w = None if rank_weights is None \
            else [rank_weights[r] for r in range(len(sel))]
        q = genre_distribution(sel, p_g_given_i, w)
        return (calibration_kl(pt, q, alpha) if metric == "kl"
                else calibration_hellinger(pt, q))

    chosen, obj = [], []
    for _ in range(Nn):
        best, bi = None, None
        for i in range(n):
            if i in chosen:
                continue
            cand = chosen + [i]
            val = (1.0 - lm) * sum(s[j] for j in cand) \
                - lm * cal(cand)
            if best is None or val > best:
                best, bi = val, i
        chosen.append(bi)
        obj.append(best)
    q_final = genre_distribution(
        chosen, p_g_given_i,
        None if rank_weights is None
        else [rank_weights[r] for r in range(len(chosen))])
    top = sorted(range(n), key=lambda i: -s[i])[:Nn]
    return RichResult(payload={
        "estimate": chosen, "ranking": chosen,
        "objective_path": obj,
        "q": q_final, "p_target": pt,
        "calibration": cal(chosen),
        "calibration_uncalibrated": cal(top),
        "score": sum(s[i] for i in chosen),
        "score_uncalibrated": sum(s[i] for i in top),
        "lambda": lm, "metric": metric, "N": Nn,
        "guarantee": "(1 - 1/e) optimal at EVERY prefix, by "
                     "submodularity",
        "method": "greedy maximum marginal relevance; Steck (2018) "
                  "eq. (6)",
    })


def cheatsheet():
    return ("caltbR: ranking by accuracy CROWDS OUT the user's "
            "minority interests -- with 70/30 genre proportions the "
            "top-10 by p(i|u) is all romance, because the imbalanced "
            "majority label is the accuracy-optimal prediction. "
            "Calibration therefore costs accuracy by construction. "
            "C_KL = KL(p || (1-alpha)q + alpha*p), alpha = 0.01, "
            "chosen because it is zero only at equality, punishes "
            "errors where p is SMALL, and prefers the less extreme "
            "deviation. Applied by greedy MMR re-ranking, submodular, "
            "so every PREFIX is (1-1/e) optimal. Not the same as "
            "diversity: diversity would return 50/50.")


# compact alias per ledger/NAMING.md
calibratedrecommendations = calibrated_rerank

# public names resolved by fn/_lazy_map.json
calibrated_rec = calibrated_rerank
calibratedrec = calibrated_rerank
