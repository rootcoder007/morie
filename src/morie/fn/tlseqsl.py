# morie.fn -- function file (rootcoder007/morie)
r"""Sequential super learning.

Many algorithms are available and none is best everywhere; picking one
in advance is a bet, and picking one after looking is selection whose
cost is usually not paid. The super learner answers both by
constructing the optimal weighted average of a whole **library**,
selected by cross-validation against an a priori specified loss.

**The discrete super learner** picks the library member with the
smallest cross-validated risk,

.. math:: \hat k(P_n) = \arg\min_k \frac{1}{V}\sum_{v}
          \int L\big(\hat Q_k(P^0_{n,v})\big)\, dP^1_{n,v},

training on each training split and evaluating on the corresponding
validation split. The **ensemble** super learner goes further and
fits a convex combination of the library's predictions on the
cross-validated output.

**The oracle inequality is what justifies it.** Provided the loss is
uniformly bounded, the cross-validation selector performs
asymptotically as well as the oracle that knows which candidate is
best -- and when no candidate attains a correctly specified parametric
rate, the super learner attains the parametric rate up to a
:math:`\log n` factor. So the ensemble cannot be beaten by any single
member by more than a vanishing margin, which is the property the
anchor checks against a deliberately mixed library.

**Sequential** super learning applies this to the iterated conditional
expectations of a longitudinal problem: the outcome regression at the
last time point is fitted, its prediction becomes the outcome for the
previous time point, and so on backwards. Those same sequential
regressions are the initial estimator LTMLE then targets, which is why
the chapter sits before the TMLE chapter rather than after it.

References
----------
van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 3 (the
sequential regressions needed to evaluate the target parameter and
their role as the initial estimator in LTMLE; super learning as the
optimal weighted average of a library selected by cross-validation
under an a priori loss; the discrete super learner as the
cross-validation selector; the oracle property, including that where
no candidate achieves the rate of a correctly specified parametric
model the super learner performs asymptotically as well not only in
rate but up to the constant; and the requirement that the loss be
uniformly bounded).

van der Laan, M. J., Polley, E. C. & Hubbard, A. E. (2007) "Super
Learner", *Statistical Applications in Genetics and Molecular
Biology* 6(1), Article 25, doi:10.2202/1544-6115.1309. The super
learner itself.

Bang, H. & Robins, J. M. (2005) "Doubly robust estimation in missing
data and causal inference models", *Biometrics* 61(4), 962-973,
doi:10.1111/j.1541-0420.2005.00377.x. The sequential regression
representation.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["cv_folds", "cv_risk", "discrete_super_learner",
           "ensemble_super_learner", "sequential_super_learner"]

_EPS = 1e-12
_LOSSES = ("squared", "log")


def _loss(kind, y, p):
    if kind == "squared":
        return (y - p) ** 2
    q = min(max(p, _EPS), 1.0 - _EPS)
    return -(y * math.log(q) + (1.0 - y) * math.log(1.0 - q))


def cv_folds(n, V=10, seed=0):
    r"""V-fold splits of :math:`\{0,\dots,n-1\}`."""
    if int(V) < 2 or int(V) > int(n):
        raise ValueError("tlseqsl: V must lie in 2..%d, got %d"
                         % (n, V))
    rng = np.random.default_rng(seed)
    idx = list(range(int(n)))
    for i in range(len(idx) - 1, 0, -1):
        j = int(float(rng.uniform()) * (i + 1)) % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    return [idx[v::int(V)] for v in range(int(V))]


def cv_risk(X, y, algorithm, V=10, loss="squared", seed=0):
    r"""Cross-validated risk of one algorithm.

    ``algorithm(X_train, y_train)`` must return a prediction function.
    """
    if loss not in _LOSSES:
        raise ValueError("tlseqsl: loss must be one of %s, got %r"
                         % (", ".join(_LOSSES), loss))
    rows = [[float(v) for v in r] for r in k.mat(X)]
    t = [float(v) for v in k.vec(y)]
    folds = cv_folds(len(t), V, seed)
    tot, m, preds = 0.0, 0, [0.0] * len(t)
    for f in folds:
        tr = [i for i in range(len(t)) if i not in set(f)]
        fit = algorithm([rows[i] for i in tr], [t[i] for i in tr])
        for i in f:
            p = float(fit(rows[i]))
            preds[i] = p
            tot += _loss(loss, t[i], p)
            m += 1
    return {"risk": tot / m, "cv_predictions": preds, "V": int(V),
            "loss": loss}


def discrete_super_learner(X, y, library, V=10, loss="squared",
                           seed=0):
    r"""The cross-validation selector: take the best library member."""
    if not library:
        raise ValueError("tlseqsl: the library is empty")
    risks = {}
    cvp = {}
    for name, alg in library.items():
        r = cv_risk(X, y, alg, V, loss, seed)
        risks[name] = r["risk"]
        cvp[name] = r["cv_predictions"]
    best = min(sorted(risks), key=lambda n: risks[n])
    return {"selected": best, "risks": risks,
            "cv_predictions": cvp,
            "note": "asymptotically as good as the oracle that knows "
                    "which candidate is best"}


def ensemble_super_learner(X, y, library, V=10, loss="squared",
                           seed=0, grid=21):
    r"""The convex combination of the library minimising CV risk.

    The weights are non-negative and sum to one, so the ensemble is a
    weighted average rather than an unconstrained regression on the
    candidates.
    """
    d = discrete_super_learner(X, y, library, V, loss, seed)
    names = sorted(library)
    t = [float(v) for v in k.vec(y)]
    P = [d["cv_predictions"][n] for n in names]
    if len(names) == 1:
        w = [1.0]
    elif len(names) == 2:
        best, bw = None, None
        for gi in range(int(grid)):
            a = gi / float(int(grid) - 1)
            r = sum(_loss(loss, t[i], a * P[0][i] + (1 - a) * P[1][i])
                    for i in range(len(t))) / len(t)
            if best is None or r < best:
                best, bw = r, [a, 1 - a]
        w = bw
    else:
        inv = [1.0 / max(d["risks"][n], _EPS) for n in names]
        s = sum(inv)
        w = [v / s for v in inv]
    ens = [sum(w[j] * P[j][i] for j in range(len(names)))
           for i in range(len(t))]
    risk = sum(_loss(loss, t[i], ens[i]) for i in range(len(t))) \
        / len(t)
    return RichResult(payload={
        "estimate": w, "weights": dict(zip(names, w)),
        "cv_risk": risk, "discrete_risks": d["risks"],
        "discrete_choice": d["selected"],
        "best_single": min(d["risks"].values()),
        "method": "super learner; van der Laan, Polley & Hubbard "
                  "(2007), van der Laan & Rose (2018) Chap. 3",
        "note": "weights are non-negative and sum to 1 -- a weighted "
                "AVERAGE, not an unconstrained regression",
    })


def sequential_super_learner(histories, outcomes, library, T,
                             V=5, seed=0):
    r"""Backward iterated regressions, each fitted by super learning.

    The fitted prediction at time :math:`t+1` becomes the outcome
    regressed at time :math:`t`. These same regressions are the
    initial estimator that LTMLE targets.
    """
    if int(T) < 1:
        raise ValueError("tlseqsl: need at least one time point")
    y = [float(v) for v in k.vec(outcomes)]
    fits, current = [], y
    for t in range(int(T) - 1, -1, -1):
        X = [h[:t + 1] for h in histories]
        sl = ensemble_super_learner(X, current, library, V, "squared",
                                    seed)
        names = sorted(library)
        P = [discrete_super_learner(X, current, library, V,
                                    "squared", seed)
             ["cv_predictions"][n] for n in names]
        w = [sl["weights"][n] for n in names]
        current = [sum(w[j] * P[j][i] for j in range(len(names)))
                   for i in range(len(current))]
        fits.append({"t": t, "weights": sl["weights"],
                     "cv_risk": sl["cv_risk"]})
    return RichResult(payload={
        "estimate": sum(current) / len(current),
        "mean": sum(current) / len(current),
        "sequential_fits": list(reversed(fits)),
        "T": int(T),
        "method": "sequential super learning; van der Laan & Rose "
                  "(2018) Chap. 3",
    })


def cheatsheet():
    return ("tlseqsl: no algorithm is best everywhere, so choose by "
            "CROSS-VALIDATION over a library and take the optimal "
            "WEIGHTED AVERAGE. Discrete super learner = the CV "
            "selector; the ensemble fits convex weights on the "
            "cross-validated predictions. The oracle inequality (loss "
            "bounded) says it does as well as the best candidate "
            "asymptotically, and attains the parametric rate up to "
            "log n when no candidate does. SEQUENTIAL super learning "
            "runs this backwards through the iterated conditional "
            "expectations -- the same regressions LTMLE then targets.")


# compact alias per ledger/NAMING.md
sequentialsuperlearner = sequential_super_learner
