# morie.fn -- function file (rootcoder007/morie)
r"""Individualized treatment rules from a causal forest.

Given an estimated conditional treatment effect :math:`\hat\tau(x)`, the
rule that maximises the mean outcome treats exactly where the effect is
positive,

.. math:: \hat d(x) = 1\{\hat\tau(x) > \eta\},

with :math:`\eta = 0` when treatment is free and :math:`\eta` the cost
per unit otherwise. The value of a rule is
:math:`V(d) = E[Y(d(X))]`, estimated by the doubly robust score

.. math:: \hat\Gamma_i = \hat\mu_{d(X_i)}(X_i)
          + \frac{1\{W_i = d(X_i)\}}{e_{W_i}(X_i)}
          \big(Y_i - \hat\mu_{W_i}(X_i)\big),

so a mistake in either the outcome model or the propensity is survivable
but a mistake in both is not.

**Evaluating a rule on the data that produced it is the trap.** The rule
is the argmax of a noisy surface, so scoring it in sample inherits the
winner's curse: a rule fitted to pure noise still looks profitable.
``evaluate="split"`` fits :math:`\hat\tau` on one half and scores on the
other, and the anchor builds a no-effect design where the in-sample
value is positive and the split-sample value is not.

References
----------
Athey, S. & Wager, S. (2021) "Policy Learning With Observational Data",
*Econometrica* 89(1), 133-161, doi:10.3982/ECTA15732. The doubly robust
scores and the regret bound for the learned rule.

Athey, S., Tibshirani, J. & Wager, S. (2019) "Generalized Random
Forests", *The Annals of Statistics* 47(2), 1148-1178,
doi:10.1214/18-AOS1709, arXiv:1610.01271. The forest that supplies
tau-hat.

Zhao, Y., Zeng, D., Rush, A. J. & Kosorok, M. R. (2012) "Estimating
Individualized Treatment Rules Using Outcome Weighted Learning",
*Journal of the American Statistical Association* 107(499), 1106-1118,
doi:10.1080/01621459.2012.695674. The value-maximisation framing.

Manski, C. F. (2004) "Statistical Treatment Rules for Heterogeneous
Populations", *Econometrica* 72(4), 1221-1246,
doi:10.1111/j.1468-0262.2004.00530.x. Treatment rules as the object of
inference.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult
from .hntfst import forest_weights, grow_forest

__all__ = ["itr_forest", "rule_value", "dr_scores", "policy_from_tau"]

_EPS = 1e-12


def policy_from_tau(tau, cost=0.0):
    """d(x) = 1{tau(x) > cost}."""
    return [1.0 if float(v) > float(cost) else 0.0 for v in tau]


def dr_scores(y, W, mu1, mu0, e, d):
    r"""The doubly robust score of a rule at each observation."""
    n = len(y)
    out = []
    for i in range(n):
        pick = d[i]
        mu = mu1[i] if pick == 1.0 else mu0[i]
        ew = e[i] if W[i] == 1.0 else 1.0 - e[i]
        if ew <= _EPS:
            raise ValueError("itrgrf: a propensity of zero at row %d"
                             % i)
        resid = 0.0
        if W[i] == pick:
            muw = mu1[i] if W[i] == 1.0 else mu0[i]
            resid = (y[i] - muw) / ew
        out.append(mu + resid)
    return out


def rule_value(y, W, mu1, mu0, e, d):
    """V(d), the mean doubly robust score, with its standard error."""
    g = dr_scores(y, W, mu1, mu0, e, d)
    n = len(g)
    v = sum(g) / n
    se = k.sd(g) / math.sqrt(n) if n > 1 else float("nan")
    return v, se, g


def _fit_arm(X, y, W, arm, rows, at_rows, n_trees, min_leaf, seed):
    """E[Y | W=arm, X], fitted on `rows` and read off at `at_rows`."""
    idx = [i for i in rows if W[i] == arm]
    if len(idx) < 4 * min_leaf:
        raise ValueError("itrgrf: too few rows in treatment arm %g"
                         % arm)
    Xa = [X[i] for i in idx]
    ya = [y[i] for i in idx]
    trees, _, _ = grow_forest(Xa, ya, n_trees=n_trees,
                              min_leaf=min_leaf, seed=seed)
    out = []
    for i in at_rows:
        w = forest_weights(trees, Xa, X[i])
        out.append(sum(w[t] * ya[t] for t in range(len(idx))))
    return out


def itr_forest(y, W, X, cost=0.0, n_trees=150, min_leaf=5, seed=0,
               evaluate="split", propensity=None, level=0.95):
    r"""Learn a treatment rule and estimate its value honestly.

    ``evaluate="split"`` learns the rule on one half and values it on
    the other. ``evaluate="in-sample"`` does both on all the data, and
    is kept because the optimism it produces is the thing to be aware
    of.
    """
    if evaluate not in ("split", "in-sample"):
        raise ValueError("itrgrf: evaluate must be split or in-sample, "
                         "got %r" % (evaluate,))
    yv, Wv = k.vec(y), k.vec(W)
    n = len(yv)
    if len(Wv) != n:
        raise ValueError("itrgrf: %d outcomes but %d treatments" % (n,
                                                                    len(Wv)))
    if any(v not in (0.0, 1.0) for v in Wv):
        raise ValueError("itrgrf: the treatment must be binary 0/1")
    Xm = k.mat(X)
    if len(Xm) != n:
        raise ValueError("itrgrf: %d covariate rows for %d outcomes"
                         % (len(Xm), n))
    if n < 60:
        raise ValueError("itrgrf: need at least 60 observations, got %d"
                         % n)
    if propensity is None:
        e = [sum(Wv) / n] * n
    else:
        e = [min(max(float(v), 1e-3), 1.0 - 1e-3) for v in propensity]
        if len(e) != n:
            raise ValueError("itrgrf: %d propensities for %d rows"
                             % (len(e), n))

    rng = np.random.default_rng(seed)
    if evaluate == "split":
        perm = sorted(range(n), key=lambda _i: float(rng.uniform()))
        learn, score = perm[: n // 2], perm[n // 2:]
    else:
        learn = score = list(range(n))

    mu1 = _fit_arm(Xm, yv, Wv, 1.0, learn, list(range(n)), n_trees,
                   min_leaf, seed)
    mu0 = _fit_arm(Xm, yv, Wv, 0.0, learn, list(range(n)), n_trees,
                   min_leaf, seed + 1)
    tau = [mu1[i] - mu0[i] for i in range(n)]
    d = policy_from_tau(tau, cost)

    ys = [yv[i] for i in score]
    Ws = [Wv[i] for i in score]
    v, se, g = rule_value(ys, Ws, [mu1[i] for i in score],
                          [mu0[i] for i in score], [e[i] for i in score],
                          [d[i] for i in score])
    v_all, se_all, _ = rule_value(ys, Ws, [mu1[i] for i in score],
                                  [mu0[i] for i in score],
                                  [e[i] for i in score],
                                  [1.0] * len(score))
    v_none, se_none, _ = rule_value(ys, Ws, [mu1[i] for i in score],
                                    [mu0[i] for i in score],
                                    [e[i] for i in score],
                                    [0.0] * len(score))
    z = k.qnorm(0.5 + 0.5 * float(level))
    return RichResult(payload={
        "estimate": v, "value": v, "se": se,
        "ci": (v - z * se, v + z * se),
        "rule": d, "tau": tau, "mu1": mu1, "mu0": mu0,
        "treated_fraction": sum(d) / n,
        "value_treat_all": v_all, "value_treat_none": v_none,
        "gain_over_treat_all": v - v_all,
        "gain_over_treat_none": v - v_none,
        "scores": g, "cost": float(cost), "evaluate": evaluate,
        "n": n, "n_scored": len(score), "level": float(level),
        "method": "individualized treatment rule from a causal forest, "
                  "valued by doubly robust scores, Athey & Wager (2021)",
    })


def cheatsheet():
    return ("itrgrf: d(x) = 1{tau(x) > cost}; value it with the doubly "
            "robust score mu_d(X) + 1{W=d}/e_W (Y - mu_W). Learn the "
            "rule and score it on DIFFERENT halves -- the rule is an "
            "argmax, so scoring it in sample inherits the winner's "
            "curse and a rule fitted to noise looks profitable.")


# compact alias per ledger/NAMING.md
itrforest = itr_forest
