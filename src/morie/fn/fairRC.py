# morie.fn -- function file (rootcoder007/morie)
r"""Measuring fairness in ranked outputs.

Fairness work in classification asks whether protected-group
membership influenced a label. Ranking is different: the output is an
*ordering*, so the question becomes whether membership influenced
**position**. A ranking has statistical parity when it did not.

**Fairness at the top matters more than fairness overall.** The
measures here are computed set-wise at discrete cut-offs -- top-10,
top-20, and so on -- and combined with a logarithmic discount, so
being unfair in the first ten costs more than being unfair in the
first hundred. The discounting is borrowed from nDCG, where the same
reasoning applies to relevance.

**Three measures, all in :math:`[0,1]`, best at 0.**

* :math:`rND` compares the protected group's share of the top-:math:`i`
  with its share of the whole population:

  .. math:: rND(\tau) = \frac{1}{Z}\sum_{i=10,20,\dots}
            \frac{1}{\log_2 i}
            \Big|\frac{|S^+_{1..i}|}{i} - \frac{|S^+|}{N}\Big|.

* :math:`rKL` replaces the absolute difference with the KL divergence
  between the two Bernoulli distributions, which is smoother and
  reacts more strongly where the proportions are small.
* :math:`rRD` compares :math:`|S^+_{1..i}|` with :math:`|S^-_{1..i}|`
  rather than with :math:`i`.

**The three are not interchangeable, and the paper is explicit about
where each breaks.** :math:`rND` and :math:`rKL` treat :math:`S^+` and
:math:`S^-` symmetrically -- under-representing *either* group at the
top scores as unfair. :math:`rRD` does not, and is therefore only
meaningful when the protected group is the minority. All three reach
their best value when the top-:math:`i` proportion matches the
population proportion, not when it is 50/50: a group that is 20% of
the population is fairly represented at 20%, not at half.

:math:`Z` is the maximum the un-normalised sum attains over all
permutations -- computed here by placing the protected group entirely
last, the worst arrangement.

References
----------
Yang, K. & Stoyanovich, J. (2017) "Measuring Fairness in Ranked
Outputs", *Proceedings of the 29th International Conference on
Scientific and Statistical Database Management (SSDBM '17)*,
doi:10.1145/3085504.3085526, arXiv:1610.08559. Sec. 3 (statistical
parity for rankings; set-based fairness computed at discrete cut-offs
with a logarithmic discount inspired by nDCG; normalisation to [0,1]
with 0 the fairest; the definitions of rND, rKL and rRD) and the
accompanying discussion of Figures 3-5 (rND and rKL treat the two
groups symmetrically and are best when the top-i proportion matches
the population proportion, while rRD is only applicable when the
protected group is the minority).

Jarvelin, K. & Kekalainen, J. (2002) "Cumulated gain-based evaluation
of IR techniques", *ACM Transactions on Information Systems* 20(4),
422-446, doi:10.1145/582415.582418. The logarithmic discount.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["rND", "rKL", "rRD", "normalizer", "cutoffs"]

_EPS = 1e-12
_MEASURES = ("rND", "rKL", "rRD")


def cutoffs(N, step=10):
    r"""The discrete evaluation points :math:`10, 20, \dots, N`."""
    n, s = int(N), int(step)
    if n < s:
        raise ValueError("fairRC: the ranking of %d is shorter than "
                         "the first cut-off %d" % (n, s))
    return list(range(s, n + 1, s))


def _shares(protected, i):
    top = protected[:i]
    return sum(top) / float(i)


def _raw(protected, measure, step):
    N = len(protected)
    P = sum(protected) / float(N)
    tot = 0.0
    for i in cutoffs(N, step):
        p = _shares(protected, i)
        w = 1.0 / math.log(i, 2)
        if measure == "rND":
            tot += w * abs(p - P)
        elif measure == "rKL":
            a = min(max(p, _EPS), 1.0 - _EPS)
            b = min(max(P, _EPS), 1.0 - _EPS)
            tot += w * (a * math.log(a / b)
                        + (1.0 - a) * math.log((1.0 - a) / (1.0 - b)))
        else:
            npos = sum(protected[:i])
            nneg = i - npos
            r1 = 0.0 if nneg == 0 or npos == 0 \
                else npos / float(nneg)
            NP, NN = sum(protected), N - sum(protected)
            r2 = 0.0 if NN == 0 or NP == 0 else NP / float(NN)
            tot += w * abs(r1 - r2)
    return tot


def normalizer(protected, measure="rND", step=10):
    r""":math:`Z`: the value of the worst arrangement.

    The protected group placed entirely last maximises the deviation
    at every cut-off.
    """
    n = len(protected)
    npos = sum(int(v) for v in protected)
    worst = [0] * (n - npos) + [1] * npos
    z = _raw(worst, measure, step)
    return z if z > _EPS else 1.0


def _measure(protected, measure, step, normalize, caveat=None):
    p = [1 if int(v) else 0 for v in protected]
    if measure not in _MEASURES:
        raise ValueError("fairRC: measure must be one of %s, got %r"
                         % (", ".join(_MEASURES), measure))
    if not p:
        raise ValueError("fairRC: the ranking is empty")
    if sum(p) in (0, len(p)):
        raise ValueError("fairRC: fairness is undefined when every "
                         "item is in one group")
    raw = _raw(p, measure, step)
    z = normalizer(p, measure, step) if normalize else 1.0
    pay = {
        "estimate": raw / z, "value": raw / z, "raw": raw,
        "normalizer": z, "measure": measure,
        "protected_share": sum(p) / float(len(p)),
        "cutoffs": cutoffs(len(p), step),
        "method": "Yang & Stoyanovich (2017) Sec. 3",
        "note": "0 is fairest; the best value is reached when the "
                "top-i share matches the POPULATION share, not 50/50",
    }
    if caveat is not None:
        pay["caveat"] = caveat
    return RichResult(payload=pay)


def rND(protected, step=10, normalize=True):
    r"""Normalised discounted difference."""
    return _measure(protected, "rND", step, normalize)


def rKL(protected, step=10, normalize=True):
    r"""Normalised discounted KL divergence."""
    return _measure(protected, "rKL", step, normalize)


def rRD(protected, step=10, normalize=True):
    r"""Normalised discounted ratio.

    Only meaningful when the protected group is the minority -- it
    does not treat the two groups symmetrically.
    """
    p = [1 if int(v) else 0 for v in protected]
    cav = None
    if sum(p) > 0.5 * len(p):
        cav = ("rRD is NOT APPLICABLE here: the protected group is "
               "the MAJORITY, and rRD does not treat the two groups "
               "symmetrically")
    return _measure(p, "rRD", step, normalize, cav)


def cheatsheet():
    return ("fairRC: statistical parity for RANKINGS -- did group "
            "membership influence POSITION. Set-based fairness at "
            "top-10, top-20, ... with a 1/log2(i) discount, so "
            "unfairness at the top costs more (the nDCG idea). rND "
            "uses |share_top_i - share_population|, rKL the KL "
            "divergence, rRD the ratio of S+ to S-. All in [0,1], 0 is "
            "fairest, and best when the top-i share matches the "
            "POPULATION share -- 20% of the population is fairly "
            "served by 20%, not 50%. rRD is asymmetric and applies "
            "only when the protected group is the minority.")


# compact alias per ledger/NAMING.md
fairranking = rND
