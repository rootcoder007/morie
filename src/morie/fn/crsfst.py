"""Cross-fitted random survival forest for a treatment effect.

The question is what treatment does to survival, and the honest answer
has to survive two problems at once. Censoring, which the survival
forest handles. And overfitting, which cross-fitting handles: a forest
that predicted the survival of a patient it was trained on would report
its own memory back as a finding, and averaging that over the sample
produces a treatment effect with no valid standard error at all.

Cross-fitting removes it by construction. Split the sample into K folds;
for each fold, train on the OTHER K-1 and predict only the held-out one.
Every observation is then predicted by a forest that never saw it, and
the resulting predictions can be averaged without the bias the
in-sample version carries. The module counts how many observations were
predicted by a forest that had seen them, and that count is always
reported: a cross-fitting implementation whose leakage is silent is
worse than none.

The effect is measured on RESTRICTED MEAN SURVIVAL TIME rather than as
a hazard ratio. RMST up to a horizon tau is the area under the survival
curve,

    RMST(tau) = integral_0^tau S(t) dt

which for a step function is an exact finite sum, not a quadrature. It
is in units of time -- "this treatment buys four months over three
years" -- so it is interpretable without a proportional-hazards
assumption that survival data routinely violates. The horizon is a
parameter and it matters: an RMST difference at one year and at five can
have opposite signs when the curves cross, and a method that hid tau
would hide that.

Two forests per fold, one per arm, so the arms are allowed different
covariate structure. Comparing them at the same covariate value gives a
conditional effect, and averaging those gives the population one.

References
  Cui, Y., Kosorok, M.R., Sverdrup, E., Wager, S. and Zhu, R. (2023)
    "Estimating heterogeneous treatment effects with right-censored data
    via causal survival forests." Journal of the Royal Statistical
    Society Series B 85(2), 179-211. doi:10.1093/jrsssb/qkac001.
  Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
    Newey, W. and Robins, J. (2018) "Double/debiased machine learning
    for treatment and structural parameters." The Econometrics Journal
    21(1), C1-C68. Cross-fitting and why the sample split is not
    optional.
  Royston, P. and Parmar, M.K.B. (2013) "Restricted mean survival time:
    an alternative to the hazard ratio." BMC Medical Research
    Methodology 13, 152. RMST as the estimand.
  Uno, H., Claggett, B., Tian, L., Inoue, E., Gallo, P., Miyata, T.,
    Schrag, D., Takeuchi, M., Uyama, Y., Zhao, L., Skali, H., Solomon,
    S., Jacobus, S., Hughes, M., Packer, M. and Wei, L.J. (2014)
    "Moving beyond the hazard ratio in quantifying the clinical
    benefit-risk of therapies." Journal of Clinical Oncology 32(22),
    2380-2385.
"""

import math

from . import _array_core as _core
from . import _w3num as _w
from .qsfrgr import survival_forest, forest_weights, weighted_km
from ._richresult import RichResult

__all__ = ["crsfst", "crs_forest", "rmst", "make_folds", "cheatsheet"]


def rmst(curve, tau):
    """Restricted mean survival time: the area under a step curve.

    Exact. The curve is one until its first step and constant between
    steps, so the integral is a finite sum of rectangles and there is
    nothing to approximate. Anything past the horizon is cut off, which
    is what "restricted" means.
    """
    tau = float(tau)
    if tau <= 0.0:
        raise ValueError("the horizon must be positive")
    area = []
    prev_t = 0.0
    prev_s = 1.0
    for t, s, _, _ in curve:
        if t >= tau:
            break
        area.append(prev_s * (t - prev_t))
        prev_t = t
        prev_s = s
    area.append(prev_s * (tau - prev_t))
    return _w.csum(area)


def make_folds(n, k, seed=0):
    """A deterministic partition of 0..n-1 into k folds.

    Shuffled by the shared generator and dealt round-robin, so the folds
    are balanced to within one and the assignment is reproducible.
    """
    k = int(k)
    if k < 2:
        raise ValueError("cross-fitting needs at least two folds")
    if k > n:
        raise ValueError("more folds than observations")
    rng = _core._SplitMix64(seed)
    idx = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(rng.uniform() * (i + 1))
        if j > i:
            j = i
        idx[i], idx[j] = idx[j], idx[i]
    fold = [0] * n
    for pos in range(n):
        fold[idx[pos]] = pos % k
    return fold


def crs_forest(time, event, D, X, K=3, tau=None, n_trees=8, min_leaf=3,
               max_depth=3, honest=True, seed=0, rule="logrank"):
    """Cross-fitted treatment effect on restricted mean survival time.

    Parameters
    ----------
    time, event : sequence
        Observed time and event indicator.
    D : sequence
        Treatment, zero or one.
    X : sequence of sequences
        Covariates.
    K : int
        Folds.
    tau : float or None
        The horizon. Defaults to the largest time observed in BOTH
        arms, because an RMST beyond the point where one arm has any
        data left is an extrapolation dressed as an estimate.

    Returns
    -------
    RichResult
        The per-observation conditional effect, the average effect, the
        two arms' restricted means, the fold assignment, and the
        leakage count -- which must be zero.

    References
    ----------
    Cui et al. (2023) JRSS-B 85(2), 179-211; Chernozhukov et al. (2018)
    Econometrics Journal 21(1), C1-C68.
    """
    t = [float(v) for v in time]
    e = [1 if v else 0 for v in event]
    d = [1 if v else 0 for v in D]
    xs = [[float(v) for v in row] for row in X]
    n = len(t)
    if len(e) != n or len(d) != n or len(xs) != n:
        raise ValueError("time, event, D and X must agree in length")
    if n < 8:
        raise ValueError("need at least eight observations to cross-fit")
    n1 = sum(d)
    if n1 == 0 or n1 == n:
        raise ValueError("both arms must be present")
    if tau is None:
        # The last time either arm can still speak for. Going past it is
        # extrapolation, and picking the overall maximum would quietly
        # do exactly that whenever one arm ends earlier.
        m1 = max(t[i] for i in range(n) if d[i])
        m0 = max(t[i] for i in range(n) if not d[i])
        tau = m1 if m1 < m0 else m0
    tau = float(tau)

    fold = make_folds(n, K, seed)
    cate = [float("nan")] * n
    r1 = [float("nan")] * n
    r0 = [float("nan")] * n
    leaked = 0
    used = 0
    for f in range(int(K)):
        tr = [i for i in range(n) if fold[i] != f]
        te = [i for i in range(n) if fold[i] == f]
        if not te:
            continue
        a1 = [i for i in tr if d[i]]
        a0 = [i for i in tr if not d[i]]
        if len(a1) < 4 or len(a0) < 4:
            continue
        for arm, rows, dest in ((1, a1, r1), (0, a0, r0)):
            sx = [xs[i] for i in rows]
            st = [t[i] for i in rows]
            se = [e[i] for i in rows]
            trees = survival_forest(sx, st, se, n_trees, None, min_leaf,
                                    max_depth, honest, seed + 100 * f + arm,
                                    rule)
            if not trees:
                continue
            for i in te:
                if i in rows:
                    leaked += 1
                w, _ = forest_weights(trees, xs[i], len(rows))
                dest[i] = rmst(weighted_km(st, se, w), tau)
        for i in te:
            if r1[i] == r1[i] and r0[i] == r0[i]:
                cate[i] = r1[i] - r0[i]
                used += 1

    got = [v for v in cate if v == v]
    if not got:
        raise ValueError("no fold produced a comparable pair of arms; "
                         "the sample is too small or too unbalanced")
    ate = _w.csum(got) / len(got)
    if len(got) > 1:
        v = _w.csum((g - ate) * (g - ate) for g in got) / (len(got) - 1)
        se = math.sqrt(v / len(got))
    else:
        se = float("nan")
    return RichResult(payload={
        "cate": cate,
        "rmst_treated": r1,
        "rmst_control": r0,
        "estimate": ate,
        "se": se,
        "ci_lower": ate - 1.959963984540054 * se if se == se else float("nan"),
        "ci_upper": ate + 1.959963984540054 * se if se == se else float("nan"),
        "fold": fold,
        "tau": tau,
        "n_scored": used,
        "n_leaked": leaked,
        "n": n,
        "n_treated": n1,
        "n_control": n - n1,
        "n_events": sum(e),
        "K": int(K),
        "honest": bool(honest),
        "rule": rule,
        "method": "cross-fitted random survival forest",
    })


crsfst = crs_forest


def cheatsheet():
    return ("crsfst: cross-fitted random survival forest. K-fold "
            "out-of-fold prediction, effect on restricted mean survival "
            "time up to tau")
