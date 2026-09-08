# morie.fn -- function file (rootcoder007/morie)
r"""Tree-based individualized treatment rules.

Two ways to choose who gets what treatment, and they fail differently.
Fit a regression of outcome on covariates and treatment, then assign
whatever the model says is best -- but a model simple enough to be
interpretable is probably misspecified, and one complex enough to be
right is unreadable. Or search directly over rules for the one with the
best estimated value -- but then the rule has no interpretable form at
all, which is why clinicians will not use it.

**Trees get both.** Search directly over rules, but restrict the search
to rules representable as a decision tree. The result is a policy that
maximises estimated value *and* prints as a handful of if-then splits.

**The value of a rule, and why a regression is not needed to get it.**
With :math:`Y^*(a)` the potential outcome under treatment :math:`a` and
:math:`\pi` a rule, the target is :math:`E[Y^*(\pi)]`. Under positivity,
strong ignorability and consistency (Assumptions 1-3), inverse
probability weighting identifies it from the data alone:

.. math:: \widehat V(\pi) = \frac{1}{n}\sum_i
          \frac{Y_i\,\mathbb 1\{A_i = \pi(X_i)\}}{p(A_i \mid X_i)}.

Only subjects whose observed treatment agrees with the rule contribute,
reweighted by how likely that agreement was. In a randomised trial with
:math:`p = 1/2` this is just twice the mean outcome among the
concordant, which the anchor checks against a closed form.

**Positivity is not a technicality.** If some treatment has probability
near zero for some covariate pattern, its weight explodes and the value
of any rule assigning it there is estimated from almost nothing.
Assumption 1 demands :math:`p(a \mid X) \ge \epsilon`; ``min_propensity``
enforces it and refuses rather than returning a number built on a
single reweighted observation.

**Purity, but for treatment allocation.** A classification tree splits
to make the response pure within nodes. Here the analogue is to split
so that the *best treatment* is homogeneous within a node: the split
score is the gain in estimated value from letting the two children
choose different treatments rather than sharing one. That is the
minimum-impurity decision assignment; each leaf then takes the
treatment maximising its own value estimate.

**The augmented alternative, and why both are here.** Pure IPW discards
every discordant observation, which is wasteful and high-variance.
Augmenting with an outcome model -- the adaptive contrast of Tao and
Wang -- uses all the data and stays consistent if either the propensity
or the outcome model is right. ``method="augmented"`` does that;
``method="ipw"`` is the unaugmented estimator. Neither dominates in
every design, so both are exposed.

References
----------
Laber, E. B. & Zhao, Y. Q. (2015) "Tree-based methods for
individualized treatment regimes", *Biometrika* 102(3), 501-514,
doi:10.1093/biomet/asv028. Sec. 2.1 (the optimal rule and Assumptions
1-3: positivity, strong ignorability, consistency), Sec. 2.2 (purity
measures for treatment allocation and the rectangular-region
representation), and the treatment of continuous treatments by kernel
smoothing.

Tao, Y. & Wang, L. (2017) "Adaptive Contrast Weighted Learning for
Multi-Stage Multi-Treatment Decision-Making", *Biometrics* 73,
145-155, doi:10.1111/biom.12539. The adaptive contrast that augments
the weighted-learning objective with an outcome model, implemented
here as ``method="augmented"``.

Zhang, B., Tsiatis, A. A., Laber, E. B. & Davidian, M. (2012) "A robust
method for estimating optimal treatment regimes", *Biometrics* 68(4),
1010-1018, doi:10.1111/j.1541-0420.2012.01763.x. The recasting of
treatment selection as a weighted classification problem that the
tree search rests on.
"""

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["rule_value", "best_treatment", "fit_tree", "predict_rule",
           "tree_rules"]

_EPS = 1e-12
_METHODS = ("ipw", "augmented")


def _check(Y, A, X, propensity, min_propensity):
    y = [float(v) for v in k.vec(Y)]
    a = [int(v) for v in k.vec(A)]
    Xm = [[float(v) for v in r] for r in k.mat(X)]
    n = len(y)
    if not (len(a) == len(Xm) == n):
        raise ValueError("trclrn: Y, A and X must agree in length "
                         "(%d, %d, %d)" % (n, len(a), len(Xm)))
    if n < 4:
        raise ValueError("trclrn: need at least 4 observations, got %d"
                         % n)
    arms = sorted(set(a))
    if len(arms) < 2:
        raise ValueError("trclrn: at least 2 treatment arms are "
                         "needed, got %d" % len(arms))
    if propensity is None:
        p = [1.0 / len(arms)] * n
    elif isinstance(propensity, (int, float)):
        p = [float(propensity)] * n
    else:
        p = [float(v) for v in k.vec(propensity)]
    if len(p) != n:
        raise ValueError("trclrn: %d propensities for %d observations"
                         % (len(p), n))
    bad = [v for v in p if v < float(min_propensity)]
    if bad:
        raise ValueError("trclrn: %d observation(s) have a propensity "
                         "below %g (smallest %.4g) -- Assumption 1 "
                         "(positivity) fails and the value of a rule "
                         "assigning that arm there is not estimable"
                         % (len(bad), min_propensity, min(bad)))
    return y, a, Xm, p, n, arms


def rule_value(Y, A, X, rule, propensity=None, method="ipw",
               outcome_model=None, min_propensity=0.01):
    r"""The estimated value :math:`\widehat V(\pi)` of a rule.

    ``rule`` maps a covariate row to a treatment label.
    ``method="ipw"`` is the inverse-probability estimator;
    ``"augmented"`` adds an outcome model and remains consistent if
    either that or the propensity is correct.
    """
    if method not in _METHODS:
        raise ValueError("trclrn: method must be ipw or augmented, "
                         "got %r" % (method,))
    y, a, Xm, p, n, arms = _check(Y, A, X, propensity, min_propensity)
    tot = 0.0
    for i in range(n):
        pi = rule(Xm[i])
        agree = 1.0 if pi == a[i] else 0.0
        if method == "ipw":
            tot += agree * y[i] / p[i]
        else:
            if outcome_model is None:
                raise ValueError("trclrn: method='augmented' needs an "
                                 "outcome_model(x, a) -> E[Y | x, a]")
            m = float(outcome_model(Xm[i], pi))
            tot += m + agree * (y[i] - m) / p[i]
    return tot / n


def best_treatment(y, a, p, rows, arms, method, Xm, outcome_model):
    """The arm maximising the estimated value within a set of rows."""
    best, bv = None, None
    for arm in arms:
        tot = 0.0
        for i in rows:
            agree = 1.0 if a[i] == arm else 0.0
            if method == "ipw":
                tot += agree * y[i] / p[i]
            else:
                m = float(outcome_model(Xm[i], arm))
                tot += m + agree * (y[i] - m) / p[i]
        if bv is None or tot > bv:
            best, bv = arm, tot
    return best, bv


def fit_tree(Y, A, X, propensity=None, method="ipw",
             outcome_model=None, max_depth=3, min_leaf=10,
             n_thresholds=20, min_propensity=0.01):
    r"""Grow a decision tree that maximises the estimated value.

    At each node the split score is the gain from letting the two
    children take different treatments rather than the node's single
    best one -- the minimum-impurity decision assignment of Sec. 2.2. A
    split that cannot improve on the parent's assignment is not made,
    so the tree stops where the data stop supporting heterogeneity.
    """
    if method not in _METHODS:
        raise ValueError("trclrn: method must be ipw or augmented, "
                         "got %r" % (method,))
    if method == "augmented" and outcome_model is None:
        raise ValueError("trclrn: method='augmented' needs an "
                         "outcome_model(x, a) -> E[Y | x, a]")
    y, a, Xm, p, n, arms = _check(Y, A, X, propensity, min_propensity)
    if int(min_leaf) < 1:
        raise ValueError("trclrn: min_leaf must be at least 1")
    d = len(Xm[0])

    def grow(rows, depth):
        arm, val = best_treatment(y, a, p, rows, arms, method, Xm,
                                  outcome_model)
        node = {"leaf": True, "treatment": arm, "n": len(rows),
                "value": val / max(len(rows), 1)}
        if depth >= int(max_depth) or len(rows) < 2 * int(min_leaf):
            return node
        best = None
        for j in range(d):
            vals = sorted({Xm[i][j] for i in rows})
            if len(vals) < 2:
                continue
            step = max(1, len(vals) // int(n_thresholds))
            for q in range(step, len(vals), step):
                thr = vals[q]
                L = [i for i in rows if Xm[i][j] < thr]
                R = [i for i in rows if Xm[i][j] >= thr]
                if len(L) < int(min_leaf) or len(R) < int(min_leaf):
                    continue
                _, vl = best_treatment(y, a, p, L, arms, method, Xm,
                                       outcome_model)
                _, vr = best_treatment(y, a, p, R, arms, method, Xm,
                                       outcome_model)
                gain = (vl + vr) - val
                if best is None or gain > best["gain"]:
                    best = {"gain": gain, "j": j, "thr": thr,
                            "L": L, "R": R}
        if best is None or best["gain"] <= _EPS:
            return node
        return {"leaf": False, "feature": best["j"],
                "threshold": best["thr"], "gain": best["gain"],
                "n": len(rows),
                "left": grow(best["L"], depth + 1),
                "right": grow(best["R"], depth + 1)}

    tree = grow(list(range(n)), 0)

    def rule(x):
        nd = tree
        while not nd["leaf"]:
            nd = (nd["left"] if float(x[nd["feature"]])
                  < nd["threshold"] else nd["right"])
        return nd["treatment"]

    v = rule_value(y, a, Xm, rule, propensity=p, method=method,
                   outcome_model=outcome_model,
                   min_propensity=min_propensity)
    fixed = {}
    for arm in arms:
        fixed[arm] = rule_value(y, a, Xm, lambda _x, _a=arm: _a,
                                propensity=p, method=method,
                                outcome_model=outcome_model,
                                min_propensity=min_propensity)
    return RichResult(payload={
        "estimate": v, "value": v, "tree": tree, "rule": rule,
        "fixed_arm_values": fixed,
        "best_fixed_arm": max(fixed, key=lambda kk: fixed[kk]),
        "n": n, "arms": arms, "method": method,
        "max_depth": int(max_depth), "min_leaf": int(min_leaf),
        "n_leaves": _count_leaves(tree),
        "method_name": "tree-based individualized treatment rule; "
                       "Laber & Zhao (2015) Sec. 2.2",
    })


def _count_leaves(nd):
    return 1 if nd["leaf"] else (_count_leaves(nd["left"])
                                 + _count_leaves(nd["right"]))


def predict_rule(tree, X):
    """Assign a treatment to each row under a fitted tree."""
    out = []
    for x in k.mat(X):
        nd = tree
        while not nd["leaf"]:
            nd = (nd["left"] if float(x[nd["feature"]])
                  < nd["threshold"] else nd["right"])
        out.append(nd["treatment"])
    return out


def tree_rules(tree, names=None, indent=0):
    """The tree as readable if-then lines -- the point of using one."""
    pad = " " * indent
    if tree["leaf"]:
        return ["%streat with %s  (n = %d)"
                % (pad, tree["treatment"], tree["n"])]
    nm = ("x%d" % tree["feature"] if names is None
          else names[tree["feature"]])
    out = ["%sif %s < %.6g:" % (pad, nm, tree["threshold"])]
    out += tree_rules(tree["left"], names, indent + 2)
    out.append("%selse:" % pad)
    out += tree_rules(tree["right"], names, indent + 2)
    return out


def cheatsheet():
    return ("trclrn: tree-based ITR. Value V(pi) = mean of Y * "
            "1{A = pi(X)} / p(A|X) -- only CONCORDANT subjects "
            "contribute, reweighted. Search over rules representable "
            "as a tree, so the winner is both value-maximal and "
            "readable. Split score = gain from letting two children "
            "choose different treatments (minimum-impurity decision "
            "assignment). Positivity is Assumption 1 and is enforced, "
            "not assumed. method='augmented' adds an outcome model "
            "(Tao & Wang 2017) and survives either model being wrong.")


# compact alias per ledger/NAMING.md
treeoptimalregime = fit_tree

# public names resolved by fn/_lazy_map.json
tree_based_regime = fit_tree
