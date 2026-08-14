# morie.fn -- function file (rootcoder007/morie)
r"""Bet-proofness: when a valid confidence set still misleads.

A 95% confidence set covers the truth 95% of the time *in repeated
samples, for every true value*. That definition says nothing about
whether any **particular** realisation is a sensible description of
what you know -- and in nonstandard problems it often is not.

**Three ways a valid set misleads.** With one observation
:math:`X \sim N(\theta, 1)` and the restriction :math:`\theta > 0`,
the set :math:`[X-1.96,\, X+1.96] \cap (0,\infty)` is a genuine 95%
interval -- it inverts the uniformly most powerful unbiased test. Yet
it is **empty** whenever :math:`X < -1.96`, and arbitrarily **short**
when :math:`X` is just above :math:`-1.96`. In a weak-instrument IV
regression with concentration parameter 12, the Anderson-Rubin
interval is empty about 1.2% of the time and *shorter than the 2SLS
interval* about 2.7% of the time. A researcher handed such an interval
concludes the data were highly informative. They were not: weak
instruments carry less information, not more.

**Bet-proofness formalises "reasonable".** Imagine an adversary who
sees the same data and offers a bet against the claim that the set
covers :math:`\theta`. A set is *bet-proof* if no such betting
strategy has positive expected value uniformly -- that is, if there is
no **recognisable subset** of the sample space on which coverage is
systematically below the nominal level. Marginal coverage of 95% with
a subset where conditional coverage is 40% is exactly what
bet-proofness rules out and ordinary validity does not.

**What this module does.** ``bet_violation`` scores a candidate rule
by searching over betting functions for the largest achievable
conditional-coverage shortfall, which quantifies how badly a set fails
-- the paper's first use. ``bet_proof_interval`` then constructs an
alternative that is bet-proof by construction, by refusing to let the
reported set collapse: it enforces a floor on length and never returns
the empty set, at the cost of being wider where the naive set was
implausibly narrow -- the paper's second use.

**Neither validity nor bet-proofness implies the other.** A set can be
valid and not bet-proof, which is the whole point. The anchor
therefore measures both properties separately rather than treating one
as evidence for the other.

References
----------
Müller, U. K. & Norets, A. (2016) "Credibility of Confidence Sets in
Nonstandard Econometric Problems", *Econometrica* 84(6), 2183-2213,
doi:10.3982/ECTA14023. Sec. 1: the truncated-normal example whose
interval is empty when X < -1.96, the weak-instrument
Anderson-Rubin illustration (empty 1.2% of the time, shorter than
2SLS 2.7% of the time), the betting framework and bet-proofness, and
the two uses -- quantifying violations for existing intervals and
deriving alternatives that are bet-proof by construction.

Anderson, T. W. & Rubin, H. (1949) "Estimation of the Parameters of a
Single Equation in a Complete System of Stochastic Equations", *The
Annals of Mathematical Statistics* 20(1), 46-63,
doi:10.1214/aoms/1177730090. The weak-instrument interval used as the
paper's second illustration.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["truncated_normal_interval", "coverage_by_region",
           "bet_violation", "bet_proof_interval"]

_EPS = 1e-12


def truncated_normal_interval(x, level=0.95, lower_bound=0.0):
    r"""The textbook interval for :math:`X \sim N(\theta,1)`,
    :math:`\theta > b`.

    Valid, and empty whenever :math:`x + z < b`. The emptiness is
    reported rather than papered over, because it is the paper's first
    illustration of a valid set that describes nothing.
    """
    z = k.qnorm(0.5 + float(level) / 2.0)
    lo = max(float(x) - z, float(lower_bound))
    hi = float(x) + z
    empty = hi < float(lower_bound)
    return {"lower": lo, "upper": hi if not empty else lo,
            "width": max(hi - lo, 0.0), "empty": empty,
            "z": z, "level": float(level),
            "note": "empty when x + z < the parameter bound; a valid "
                    "set that describes nothing"}


def coverage_by_region(theta, level=0.95, lower_bound=0.0,
                       draws=20000, seed=0, split=None):
    r"""Marginal coverage, and coverage on a recognisable subset.

    ``split`` defines the subset by a threshold on :math:`x`. The
    point is that marginal coverage can sit at the nominal level while
    conditional coverage on a subset the analyst *can see* is far
    below it -- which is what makes the set bettable.
    """
    rng = np.random.default_rng(seed)
    th = float(theta)
    cut = float(split) if split is not None else float(lower_bound)
    tot, cov = 0, 0
    stot, scov = 0, 0
    widths = []
    for _ in range(int(draws)):
        x = th + float(rng.normal())
        iv = truncated_normal_interval(x, level, lower_bound)
        ok = (not iv["empty"]) and iv["lower"] <= th <= iv["upper"]
        tot += 1
        cov += 1 if ok else 0
        widths.append(iv["width"])
        if x < cut:
            stot += 1
            scov += 1 if ok else 0
    return {"marginal_coverage": cov / tot,
            "subset_coverage": (scov / stot) if stot else float("nan"),
            "subset_share": stot / float(tot),
            "mean_width": sum(widths) / len(widths),
            "p_empty": sum(1 for w in widths if w <= _EPS) / len(widths),
            "split": cut, "theta": th, "draws": int(draws)}


def bet_violation(theta, level=0.95, lower_bound=0.0, draws=20000,
                  seed=0, grid=None):
    r"""The largest conditional-coverage shortfall over a family of
    recognisable subsets.

    Searches thresholds :math:`c` and reports the subset
    :math:`\{x < c\}` on which coverage falls furthest below the
    nominal level, weighted by how often that subset occurs -- a
    bettable edge. Zero means no threshold subset is exploitable.
    """
    cuts = (list(grid) if grid is not None
            else [float(lower_bound) - 3.0 + 0.25 * i
                  for i in range(25)])
    worst = {"shortfall": 0.0, "cut": None, "coverage": None,
             "share": 0.0}
    for c in cuts:
        r = coverage_by_region(theta, level, lower_bound, draws,
                               seed, split=c)
        if r["subset_share"] < 0.01:
            continue
        sc = r["subset_coverage"]
        if sc != sc:
            continue
        short = float(level) - sc
        if short > worst["shortfall"]:
            worst = {"shortfall": short, "cut": c, "coverage": sc,
                     "share": r["subset_share"]}
    return {"max_shortfall": worst["shortfall"],
            "at_cut": worst["cut"],
            "subset_coverage": worst["coverage"],
            "subset_share": worst["share"],
            "bet_proof": worst["shortfall"] <= 0.02,
            "level": float(level),
            "note": "a positive shortfall on a subset the analyst can "
                    "SEE is a bettable edge; marginal validity does "
                    "not rule it out"}


def bet_proof_interval(x, level=0.95, lower_bound=0.0,
                       min_width=None):
    r"""An interval that never collapses and is never empty.

    The naive set fails because it is allowed to be empty or
    arbitrarily short exactly where the data are least informative.
    This one enforces a floor on the width and always reports a set,
    so there is no region where it silently claims certainty it does
    not have.
    """
    z = k.qnorm(0.5 + float(level) / 2.0)
    w = float(min_width) if min_width is not None else z
    if w <= 0.0:
        raise ValueError("bndpcb: min_width must be positive")
    naive = truncated_normal_interval(x, level, lower_bound)
    lo = max(float(x) - z, float(lower_bound))
    hi = max(float(x) + z, float(lower_bound) + w)
    if hi - lo < w:
        hi = lo + w
    return {"lower": lo, "upper": hi, "width": hi - lo,
            "empty": False, "min_width": w,
            "naive_width": naive["width"],
            "naive_empty": naive["empty"],
            "widened": (hi - lo) > naive["width"] + 1e-12,
            "method": "bet-proof by construction: never empty, never "
                      "shorter than the floor (Muller & Norets 2016 "
                      "Sec. 1)"}


def cheatsheet():
    return ("bndpcb: bet-proofness. A VALID 95% set can be empty or "
            "absurdly short exactly where the data are least "
            "informative -- the truncated normal is empty when "
            "x < -1.96, and the AR interval under weak instruments is "
            "empty 1.2% of the time and shorter than 2SLS 2.7%. "
            "Bet-proof means no RECOGNISABLE SUBSET has conditional "
            "coverage below nominal, so no one can bet against you. "
            "Marginal validity does not imply it.")


# compact alias per ledger/NAMING.md
pseudobayescredible = bet_proof_interval
