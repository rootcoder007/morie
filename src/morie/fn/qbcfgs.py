"""Quantile-balanced score for forests: strata-local balancing weights.

Inverse-propensity weighting has one failure mode that dominates all
others: a single observation with a propensity near zero or one gets a
weight near infinity and quietly becomes the entire estimate. The usual
patches -- trimming, truncation -- throw away exactly the observations
the design was least able to match, which is honest about the symptom
and silent about the cause.

Balancing within QUANTILE STRATA is the alternative. Sort the fitted
propensities, cut them into strata of equal size, and normalise the
weights inside each stratum rather than across the whole sample. An
extreme propensity is then extreme only relative to its own stratum,
where every other unit is extreme too, so it can carry at most its
stratum's share of the estimate. Nothing is discarded and no unit can
run away with the answer.

What comes out is a per-stratum effect, a size-weighted overall effect
that is exactly the average of them, and -- the part that matters -- a
BALANCE table. The claim a weighting scheme makes is that after
weighting the arms look alike on the covariates, and the standardised
mean difference before and after is how you check it rather than take
it on faith. A weighting that did not improve balance would have no
argument for itself, and this module reports both numbers so that
argument can be lost.

The propensity comes from an honest forest, so the fitted value at a
point was not chosen using that point's own treatment. Without honesty
the propensities are shrunk towards the observed treatment and the
weights look better balanced than they are.

References
  Hsu, Y.-C., Huber, M., Lee, Y.-Y. and Pipoz, L. (2022) "Direct and
    indirect effects of continuous treatments based on generalized
    propensity score weighting." Journal of Applied Econometrics 37(2),
    449-460. Propensity-score weighting with the balance diagnostics
    this reports.
  Rosenbaum, P.R. and Rubin, D.B. (1983) "The central role of the
    propensity score in observational studies for causal effects."
    Biometrika 70(1), 41-55. Stratification on the propensity score.
  Rosenbaum, P.R. and Rubin, D.B. (1984) "Reducing bias in
    observational studies using subclassification on the propensity
    score." Journal of the American Statistical Association 79(387),
    516-524. The equal-size quantile strata used here.
  Austin, P.C. (2009) "Balance diagnostics for comparing the
    distribution of baseline covariates between treatment groups in
    propensity-score matched samples." Statistics in Medicine 28(25),
    3083-3107. The standardised mean difference.
  Wager, S. and Athey, S. (2018) "Estimation and inference of
    heterogeneous treatment effects using random forests." JASA
    113(523), 1228-1242. Honesty.
"""

import math

from . import _array_core as _core
from . import _w3num as _w
from .sdcfst import honest_forest, forest_predict
from ._richresult import RichResult

__all__ = ["qbcfgs", "qb_cf_score", "strata_of", "smd", "WEIGHTS",
           "cheatsheet"]

WEIGHTS = ("ate", "att")


def strata_of(e, n_strata):
    """Equal-size strata by the rank of the fitted propensity.

    Ranks rather than values, so a propensity distribution piled up at
    one end still gives strata of equal size -- which is the point of
    stratifying by quantile rather than by a grid.
    """
    n = len(e)
    order = sorted(range(n), key=lambda i: (e[i], i))
    s = [0] * n
    for pos in range(n):
        k = int(pos * n_strata / n)
        if k >= n_strata:
            k = n_strata - 1
        s[order[pos]] = k
    return s


def smd(x, d, w=None):
    """Standardised mean difference of one covariate between arms.

    The difference in means over the pooled standard deviation, so it
    is unit-free and comparable across covariates. With weights it is
    the weighted version, which is the number a weighting scheme is
    claiming to shrink.
    """
    n = len(x)
    if w is None:
        w = [1.0] * n
    s1 = _w.csum(w[i] for i in range(n) if d[i])
    s0 = _w.csum(w[i] for i in range(n) if not d[i])
    if s1 <= 0.0 or s0 <= 0.0:
        return float("nan")
    m1 = _w.csum(w[i] * x[i] for i in range(n) if d[i]) / s1
    m0 = _w.csum(w[i] * x[i] for i in range(n) if not d[i]) / s0
    v1 = _w.csum(w[i] * (x[i] - m1) * (x[i] - m1)
                 for i in range(n) if d[i]) / s1
    v0 = _w.csum(w[i] * (x[i] - m0) * (x[i] - m0)
                 for i in range(n) if not d[i]) / s0
    pool = 0.5 * (v1 + v0)
    if pool <= 0.0:
        return 0.0 if m1 == m0 else float("inf")
    return (m1 - m0) / math.sqrt(pool)


def qb_cf_score(y, D, X, quantile=0.5, n_strata=4, weight="ate",
                n_trees=8, min_leaf=3, max_depth=3, seed=0, clip=0.01):
    """A stratum-balanced treatment effect and its balance table.

    Parameters
    ----------
    y, D : sequence
        Outcome and binary treatment.
    X : sequence of sequences
        Covariates.
    quantile : float
        Which propensity quantile the focal stratum is read at.
    n_strata : int
        Number of equal-size strata.
    weight : str
        "ate" weights each unit towards the whole population, "att"
        towards the treated.
    clip : float
        The propensity is held inside [clip, 1 - clip] before it enters
        a denominator. Reported, not silent.

    Returns
    -------
    RichResult
        The per-stratum effects and sizes, the overall effect, the
        balance before and after weighting, and the focal stratum.

    References
    ----------
    Rosenbaum and Rubin (1984) JASA 79(387), 516-524; Austin (2009)
    Stat Med 28(25), 3083-3107.
    """
    if weight not in WEIGHTS:
        raise ValueError("weight must be one of %r" % (WEIGHTS,))
    ys = [float(v) for v in y]
    d = [1 if v else 0 for v in D]
    xs = [[float(v) for v in row] for row in X]
    n = len(ys)
    p = len(xs[0])
    if len(d) != n or len(xs) != n:
        raise ValueError("y, D and X must agree in length")
    q = int(n_strata)
    if q < 2:
        raise ValueError("need at least two strata")
    if n < 4 * q:
        raise ValueError("need at least four observations per stratum")
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("the quantile must lie strictly inside (0, 1)")
    if not 0.0 < float(clip) < 0.5:
        raise ValueError("the clip must lie strictly inside (0, 0.5)")
    rows = list(range(n))

    rng = _core._SplitMix64(seed)
    fe = honest_forest(xs, [float(v) for v in d], rows, n_trees, None,
                       min_leaf, max_depth, seed, rng)
    raw = [forest_predict(fe, xs[i]) for i in range(n)]
    n_clipped = 0
    e = []
    for v in raw:
        if v < clip:
            e.append(float(clip))
            n_clipped += 1
        elif v > 1.0 - clip:
            e.append(1.0 - float(clip))
            n_clipped += 1
        else:
            e.append(v)

    s = strata_of(e, q)
    # Balancing weights, normalised WITHIN the stratum. That single
    # word is the method: the same weights normalised globally let one
    # extreme propensity dominate everything.
    w = [0.0] * n
    for k in range(q):
        mem = [i for i in range(n) if s[i] == k]
        raww = []
        for i in mem:
            if weight == "ate":
                raww.append(1.0 / e[i] if d[i] else 1.0 / (1.0 - e[i]))
            else:
                raww.append(1.0 if d[i] else e[i] / (1.0 - e[i]))
        tot = _w.csum(raww)
        if tot <= 0.0:
            continue
        scale = len(mem) / tot
        for j, i in enumerate(mem):
            w[i] = raww[j] * scale

    eff = []
    size = []
    for k in range(q):
        mem = [i for i in range(n) if s[i] == k]
        t = [i for i in mem if d[i]]
        c = [i for i in mem if not d[i]]
        size.append(len(mem))
        if not t or not c:
            eff.append(float("nan"))
            continue
        w1 = _w.csum(w[i] for i in t)
        w0 = _w.csum(w[i] for i in c)
        if w1 <= 0.0 or w0 <= 0.0:
            eff.append(float("nan"))
            continue
        m1 = _w.csum(w[i] * ys[i] for i in t) / w1
        m0 = _w.csum(w[i] * ys[i] for i in c) / w0
        eff.append(m1 - m0)

    live = [k for k in range(q) if eff[k] == eff[k]]
    if not live:
        raise ValueError("no stratum contains both arms; reduce the "
                         "number of strata")
    tot = _w.csum(float(size[k]) for k in live)
    overall = _w.csum(eff[k] * size[k] for k in live) / tot

    before = [smd([xs[i][j] for i in range(n)], d) for j in range(p)]
    after = [smd([xs[i][j] for i in range(n)], d, w) for j in range(p)]
    fb = _w.csum(abs(v) for v in before) / p
    fa = _w.csum(abs(v) for v in after) / p

    focal = int(float(quantile) * q)
    if focal >= q:
        focal = q - 1
    return RichResult(payload={
        "stratum_effect": eff,
        "stratum_size": size,
        "stratum": s,
        "propensity": e,
        "weight_value": w,
        "smd_before": before,
        "smd_after": after,
        "mean_abs_smd_before": fb,
        "mean_abs_smd_after": fa,
        "balance_improved": fa < fb,
        "estimate": overall,
        "se": float("nan"),
        "focal_stratum": focal,
        "focal_effect": eff[focal],
        # A stratum can hold only one arm, in which case it has no
        # effect to report and its entry is not-a-number. Returning that
        # bare would make "the focal stratum is empty of controls" look
        # exactly like a numerical accident, so the deadness is a flag
        # the caller can branch on rather than a value they have to test
        # for nan-ness to discover.
        "focal_live": eff[focal] == eff[focal],
        "n_clipped": n_clipped,
        "n_live_strata": len(live),
        "n": n,
        "n_treated": sum(d),
        "n_strata": q,
        "quantile": float(quantile),
        "clip": float(clip),
        "weighting": weight,
        "method": "quantile-balanced score for forests",
    })


qbcfgs = qb_cf_score


def cheatsheet():
    return ("qbcfgs: quantile-balanced score for forests. weightings "
            + ", ".join(WEIGHTS)
            + "; balancing weights normalised within propensity strata")


# Catalogue aliases (src/morie/fn/_lazy_map.json resolves these by name).
qbcfscore = qb_cf_score
