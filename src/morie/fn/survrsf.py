# morie.fn -- function file (rootcoder007/morie)
r"""Random survival forests.

**The algorithm**, in the paper's own five steps: draw :math:`B`
bootstrap samples (each leaving out about 37% of the data, the
out-of-bag cases); grow a survival tree on each, choosing at every node
:math:`p` candidate variables at random and splitting on the one that
maximises survival difference between the daughters; grow to full size
subject to every terminal node holding at least :math:`d_0` unique
deaths; take the Nelson-Aalen estimator in each terminal node and
average over trees for the ensemble cumulative hazard; and estimate
prediction error on the out-of-bag data.

**The terminal node estimator.** For a terminal node :math:`h` with
distinct event times :math:`t_{1,h} < \cdots < t_{N(h),h}`, deaths
:math:`d_{l,h}` and at-risk counts :math:`Y_{l,h}`,

.. math:: \hat H_h(t) = \sum_{t_{l,h} \le t} \frac{d_{l,h}}{Y_{l,h}},

and every case falling in :math:`h` carries that same hazard.

**Conservation of events** is what makes the ensemble interpretable,
and it is an exact identity rather than an approximation. Lemma 1: for
each terminal node,

.. math:: \sum_{i=1}^{n(h)} \hat H_h(T_{i,h})
          = \sum_{i=1}^{n(h)} \delta_{i,h},

so the estimated hazard summed over the observed times -- censored
times included -- returns the number of deaths in the node exactly.
Corollary 1 lifts this to the whole tree. ``conservation_check``
computes both sides; the anchor turns on their difference being zero to
machine precision, which fails immediately if the at-risk set is built
wrong or a censored time is dropped.

**Mortality**, the predicted outcome, is
:math:`\hat M_i = \sum_j H_e(T_j \mid x_i)`: the number of deaths
expected if every other case behaved like :math:`i`. It is on the scale
of a count of deaths, not a probability, which is why the anchor checks
it against the total number of events rather than against 1.

**Prediction error** is :math:`1 - C`, with Harrell's concordance index
computed by the paper's four-step recipe -- form all pairs; drop a pair
whose shorter time is censored, and drop tied times unless at least one
is a death; score 1, 0.5 or 0 by the stated rules; divide concordance
by the permissible count. A value of 0.5 is guessing.

**Splitting rules.** The paper lists four. Two of them are fully
specified by sources in hand and are implemented:

``logrank``
    Maximise the two-sample log-rank statistic over the candidate
    variable and split point.
``logrankrandom``
    One random split point per candidate variable; the variable with
    the largest log-rank statistic at its random point wins.
``logrankscore``
    Hothorn and Lausen's standardised statistic on Lausen's log-rank
    scores: the scores are computed once per node, the statistic for a
    candidate cutpoint is the sum of scores on one side, standardised
    by moments *conditional on those scores*, and the split maximises
    its absolute value. Without censoring the scores reduce to Savage
    scores and sum to zero.

The remaining rule, ``conserve`` (daughters closest to the
conservation-of-events principle), is named here and refused with the
citation rather than guessed at: the paper does not define it and
points to Ishwaran & Kogalur (2008) for the details. ``SPLIT_RULES``
lists all four and ``rule_status`` says which are available and why.

**Variable importance** follows Sec. 7: drop the out-of-bag cases down
their in-bag tree and, whenever a split on :math:`x` is met, send the
case to a daughter at random; VIMP is the resulting prediction error
minus the original. Large values mark predictive variables; zero or
negative ones mark variables to filter. As the paper warns, this is
*not* the change in error from regrowing the forest without :math:`x`.

References
----------
Ishwaran, H., Kogalur, U. B., Blackstone, E. H. & Lauer, M. S. (2008)
"Random Survival Forests", *The Annals of Applied Statistics* 2(3),
841-860, doi:10.1214/08-AOAS169. Sec. 2 for the five-step algorithm and
the 37% out-of-bag fraction; Sec. 3.1-3.2 for the binary survival tree,
the :math:`d_0` terminal-node constraint and the Nelson-Aalen terminal
estimator (3.1); Sec. 3.3 for the bootstrap and out-of-bag ensembles
(3.2)-(3.3); Sec. 4, Lemma 1 and Corollary 1, for conservation of
events; Sec. 4.1 for ensemble mortality; Sec. 5.1 for the four-step
C-index calculation and Sec. 5.2 for out-of-bag prediction error
:math:`1 - C^{**}`; Sec. 6 for the four splitting rules; and Sec. 7 for
variable importance by random daughter assignment.

Harrell, F., Califf, R., Pryor, D., Lee, K. & Rosati, R. (1982)
"Evaluating the Yield of Medical Tests", *JAMA* 247(18), 2543-2546,
doi:10.1001/jama.1982.03320430047030, for the concordance index the
above recipe computes.

Hothorn, T. & Lausen, B. (2003) "On the exact distribution of
maximally selected rank statistics", *Computational Statistics & Data
Analysis* 43(2), 121-137, doi:10.1016/S0167-9473(02)00225-6.
Equations (1)-(4) for the linear rank statistic at a cutpoint, its
conditional expectation and variance and the standardised form, and
Sec. 5 equation (13) for Lausen's log-rank scores under right
censoring, including the remark that without censoring or ties they
are Savage scores and that the moments are conditional on the scores.

Segal, M. R. (1988) "Regression Trees for Censored Data", *Biometrics*
44(1), 35-47, doi:10.2307/2531894, and LeBlanc, M. & Crowley, J. (1993)
"Survival Trees by Goodness of Split", *Journal of the American
Statistical Association* 88(422), 457-467,
doi:10.1080/01621459.1993.10476296, for log-rank splitting.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["nelson_aalen", "logrank_statistic",
           "logrank_scores", "logrank_score_statistic",
           "best_split",
           "grow_tree", "predict_tree", "forest", "ensemble_chf",
           "mortality", "c_index", "conservation_check", "vimp",
           "rule_status", "SPLIT_RULES"]

SPLIT_RULES = ("logrank", "logrankrandom", "conserve", "logrankscore")
_AVAILABLE = ("logrank", "logrankrandom", "logrankscore")
_UNSOURCED = {
    "conserve": "the conservation-of-events splitting rule is named "
                "but not defined in Ishwaran et al. (2008); see "
                "Ishwaran & Kogalur (2008), the randomSurvivalForest "
                "technical manual",
}


def rule_status(rule=None):
    r"""Which of the paper's four splitting rules are implemented."""
    if rule is None:
        return {"rules": SPLIT_RULES, "available": _AVAILABLE,
                "unavailable": dict(_UNSOURCED)}
    if rule not in SPLIT_RULES:
        raise ValueError("survrsf: rule must be one of %s, got %r"
                         % (", ".join(SPLIT_RULES), rule))
    return {"rule": rule, "available": rule in _AVAILABLE,
            "reason": _UNSOURCED.get(rule, "")}


def _check_rule(rule):
    if rule not in SPLIT_RULES:
        raise ValueError("survrsf: rule must be one of %s, got %r"
                         % (", ".join(SPLIT_RULES), rule))
    if rule not in _AVAILABLE:
        raise ValueError("survrsf: the %r splitting rule is not "
                         "implemented -- %s" % (rule, _UNSOURCED[rule]))


class _Rng:
    """Small deterministic generator (no external imports here)."""

    def __init__(self, seed=0):
        self.s = (int(seed) * 6364136223846793005 + 1442695040888963407)
        self.s &= (1 << 64) - 1

    def next(self):
        self.s = (self.s * 6364136223846793005
                  + 1442695040888963407) & ((1 << 64) - 1)
        return (self.s >> 11) / float(1 << 53)

    def randint(self, n):
        return int(self.next() * n) % n

    def sample(self, seq, k):
        pool = list(seq)
        out = []
        for _ in range(min(k, len(pool))):
            out.append(pool.pop(self.randint(len(pool))))
        return out


def nelson_aalen(times, events):
    r"""The terminal-node estimator of equation (3.1)."""
    n = len(times)
    if n != len(events):
        raise ValueError("survrsf: %d times but %d event indicators"
                         % (n, len(events)))
    if n == 0:
        raise ValueError("survrsf: no observations")
    order = sorted(range(n), key=lambda i: times[i])
    ts, ds = [], []
    cum = 0.0
    i = 0
    while i < n:
        t = times[order[i]]
        j = i
        d = 0
        while j < n and times[order[j]] == t:
            d += int(events[order[j]] != 0)
            j += 1
        at_risk = n - i
        if d:
            cum += d / float(at_risk)
            ts.append(float(t))
            ds.append(cum)
        i = j
    return {"time": ts, "chf": ds, "n": n,
            "deaths": int(sum(1 for e in events if e))}


def _chf_at(na, t):
    out = 0.0
    for i, ti in enumerate(na["time"]):
        if ti <= t:
            out = na["chf"][i]
        else:
            break
    return out


def conservation_check(times, events):
    r"""Lemma 1: the hazard summed over observed times *is* the deaths.

    Censored times count too -- dropping them is the usual way to get
    this wrong, and it shows up here immediately.
    """
    na = nelson_aalen(times, events)
    total = sum(_chf_at(na, t) for t in times)
    deaths = float(sum(1 for e in events if e))
    return {"sum_chf": total, "deaths": deaths,
            "difference": total - deaths,
            "conserved": abs(total - deaths) < 1e-9}


def logrank_statistic(times, events, group):
    r"""The two-sample log-rank statistic used for splitting."""
    n = len(times)
    if not (n == len(events) == len(group)):
        raise ValueError("survrsf: times, events and group must have "
                         "the same length")
    order = sorted(range(n), key=lambda i: times[i])
    num = 0.0
    var = 0.0
    i = 0
    while i < n:
        t = times[order[i]]
        j = i
        d = d1 = 0
        while j < n and times[order[j]] == t:
            if events[order[j]]:
                d += 1
                if group[order[j]]:
                    d1 += 1
            j += 1
        at_risk = n - i
        r1 = sum(1 for k in range(i, n) if group[order[k]])
        if d and at_risk > 1:
            num += d1 - d * r1 / float(at_risk)
            var += (d * (r1 / float(at_risk))
                    * (1.0 - r1 / float(at_risk))
                    * (at_risk - d) / float(at_risk - 1))
        elif d:
            num += d1 - d * r1 / float(at_risk)
        i = j
    if var <= 0.0:
        return 0.0
    return abs(num) / math.sqrt(var)


def logrank_scores(times, events):
    r"""Lausen's log-rank scores, Hothorn & Lausen (2003) eq. (13).

    .. math:: a_i = \delta_i - \sum_{j:\ Z_j \le Z_i}
              \frac{\delta_j}{N - \gamma_j(Z) + 1},
              \qquad \gamma_j(Z) = \#\{i : Z_i \le Z_j\}.

    Without censoring or ties these are the Savage scores, and they
    sum to zero -- an exact identity the anchor uses.
    """
    N = len(times)
    if N != len(events):
        raise ValueError("survrsf: times and events must have the "
                         "same length")
    if N == 0:
        raise ValueError("survrsf: no observations")
    gamma = [sum(1 for t in times if t <= times[j]) for j in range(N)]
    out = []
    for i in range(N):
        s = 0.0
        for j in range(N):
            if times[j] <= times[i] and events[j]:
                s += 1.0 / (N - gamma[j] + 1)
        out.append(float(int(events[i] != 0)) - s)
    return out


def logrank_score_statistic(times, events, group, scores=None):
    r"""The standardised statistic of Hothorn & Lausen (2003) eqs.
    (1)-(4).

    :math:`T = \sum_{i \in \text{left}} a_i` with

    .. math:: E(T\mid X) = \frac{m}{N}\sum_i a_i, \qquad
              \mathrm{Var}(T\mid X) = \frac{m\,n}{N^2(N-1)}
              \Big(N\sum_i a_i^2 - \big(\sum_i a_i\big)^2\Big),

    and :math:`S = (T - E)/\sqrt{\mathrm{Var}}`, standard normal
    under the null. Unlike the plain log-rank statistic, the moments
    here are *conditional on the scores*, which is what makes the
    split comparable across candidate cutpoints.
    """
    N = len(times)
    a = logrank_scores(times, events) if scores is None else list(scores)
    if not (N == len(group) == len(a)):
        raise ValueError("survrsf: times, group and scores must have "
                         "the same length")
    m = sum(1 for g in group if not g)
    n = N - m
    if m == 0 or n == 0 or N < 2:
        return 0.0
    T = sum(a[i] for i in range(N) if not group[i])
    sa = sum(a)
    sa2 = sum(v * v for v in a)
    ET = m * sa / float(N)
    VT = (m * n / float(N * N * (N - 1))) * (N * sa2 - sa * sa)
    if VT <= 0.0:
        return 0.0
    return abs(T - ET) / math.sqrt(VT)


def best_split(X, times, events, features, min_deaths=3,
               rule="logrank", rng=None):
    r"""Search the candidate variables for the best split."""
    _check_rule(rule)
    n = len(times)
    best = None
    scores = (logrank_scores(times, events)
              if rule == "logrankscore" else None)
    for j in features:
        vals = sorted(set(X[i][j] for i in range(n)))
        if len(vals) < 2:
            continue
        cuts = [(vals[k] + vals[k + 1]) / 2.0
                for k in range(len(vals) - 1)]
        if rule == "logrankrandom":
            if rng is None:
                rng = _Rng(0)
            cuts = [cuts[rng.randint(len(cuts))]]
        for c in cuts:
            grp = [1 if X[i][j] > c else 0 for i in range(n)]
            left = [i for i in range(n) if not grp[i]]
            right = [i for i in range(n) if grp[i]]
            if (sum(1 for i in left if events[i]) < min_deaths
                    or sum(1 for i in right if events[i])
                    < min_deaths):
                continue
            if rule == "logrankscore":
                stat = logrank_score_statistic(times, events, grp,
                                               scores)
            else:
                stat = logrank_statistic(times, events, grp)
            if best is None or stat > best["statistic"]:
                best = {"variable": j, "cut": c, "statistic": stat,
                        "left": left, "right": right}
    return best


def grow_tree(X, times, events, mtry=None, min_deaths=3,
              rule="logrank", seed=0, rng=None):
    r"""One survival tree, grown to saturation under :math:`d_0`."""
    _check_rule(rule)
    n = len(times)
    if n == 0:
        raise ValueError("survrsf: no observations")
    d = len(X[0])
    mtry = int(mtry) if mtry else max(1, int(math.sqrt(d)))
    if rng is None:
        rng = _Rng(seed)

    def build(idx, depth):
        t = [times[i] for i in idx]
        e = [events[i] for i in idx]
        if sum(1 for v in e if v) < 2 * min_deaths or depth > 40:
            return {"leaf": True, "na": nelson_aalen(t, e),
                    "n": len(idx), "idx": list(idx)}
        feats = rng.sample(range(d), mtry)
        sub = [[X[i][j] for j in range(d)] for i in idx]
        sp = best_split(sub, t, e, feats, min_deaths, rule, rng)
        if sp is None:
            return {"leaf": True, "na": nelson_aalen(t, e),
                    "n": len(idx), "idx": list(idx)}
        left = [idx[i] for i in sp["left"]]
        right = [idx[i] for i in sp["right"]]
        return {"leaf": False, "variable": sp["variable"],
                "cut": sp["cut"], "statistic": sp["statistic"],
                "left": build(left, depth + 1),
                "right": build(right, depth + 1)}

    return {"root": build(list(range(n)), 0), "rule": rule,
            "mtry": mtry, "min_deaths": int(min_deaths), "n": n}


def _leaves(node, out=None):
    out = [] if out is None else out
    if node["leaf"]:
        out.append(node)
    else:
        _leaves(node["left"], out)
        _leaves(node["right"], out)
    return out


def predict_tree(tree, x, random_variable=None, rng=None):
    r"""Drop a case down the tree and return its terminal node.

    ``random_variable`` implements the Sec. 7 importance device: at any
    split on that variable, the daughter is chosen at random.
    """
    node = tree["root"]
    while not node["leaf"]:
        if random_variable is not None \
                and node["variable"] == random_variable:
            go_right = (rng or _Rng(0)).next() < 0.5
        else:
            go_right = x[node["variable"]] > node["cut"]
        node = node["right"] if go_right else node["left"]
    return node


def forest(X, times, events, n_trees=50, mtry=None, min_deaths=3,
           rule="logrank", seed=0):
    r"""Grow the forest, keeping the out-of-bag membership."""
    _check_rule(rule)
    n = len(times)
    rng = _Rng(seed)
    trees, inbag = [], []
    for b in range(int(n_trees)):
        boot = [rng.randint(n) for _ in range(n)]
        used = set(boot)
        Xb = [X[i] for i in boot]
        tb = [times[i] for i in boot]
        eb = [events[i] for i in boot]
        if sum(1 for v in eb if v) < 2 * min_deaths:
            continue
        trees.append(grow_tree(Xb, tb, eb, mtry, min_deaths, rule,
                               rng=rng))
        inbag.append(used)
    if not trees:
        raise ValueError("survrsf: no tree could be grown; the data "
                         "hold too few deaths for min_deaths = %d"
                         % min_deaths)
    oob_fraction = (sum(n - len(u) for u in inbag)
                    / float(len(inbag) * n))
    return {"trees": trees, "inbag": inbag, "n": n,
            "rule": rule, "n_trees": len(trees),
            "oob_fraction": oob_fraction,
            "times": list(times), "events": list(events)}


def ensemble_chf(fit, X, t, oob=True, random_variable=None, seed=1):
    r"""Equations (3.2) and (3.3): the out-of-bag or bootstrap
    ensemble."""
    rng = _Rng(seed)
    out = []
    for i in range(len(X)):
        vals, count = 0.0, 0
        for b, tree in enumerate(fit["trees"]):
            if oob and i in fit["inbag"][b]:
                continue
            node = predict_tree(tree, X[i], random_variable, rng)
            vals += _chf_at(node["na"], t)
            count += 1
        out.append(vals / count if count else float("nan"))
    return out


def mortality(fit, X, oob=True, random_variable=None, seed=1):
    r"""Sec. 4.1: the hazard summed over every observed time."""
    rng = _Rng(seed)
    ts = fit["times"]
    out = []
    for i in range(len(X)):
        total, count = 0.0, 0
        for b, tree in enumerate(fit["trees"]):
            if oob and i in fit["inbag"][b]:
                continue
            node = predict_tree(tree, X[i], random_variable, rng)
            total += sum(_chf_at(node["na"], t) for t in ts)
            count += 1
        out.append(total / count if count else float("nan"))
    return out


def c_index(times, events, predicted):
    r"""Harrell's C by the paper's four steps.

    ``predicted`` is a worse-outcome score: larger means the case is
    expected to fail sooner, which is the direction mortality runs in.
    """
    n = len(times)
    if not (n == len(events) == len(predicted)):
        raise ValueError("survrsf: times, events and predictions must "
                         "have the same length")
    permissible = 0.0
    concordance = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = times[i], times[j]
            ei, ej = events[i], events[j]
            if ti < tj and not ei:
                continue
            if tj < ti and not ej:
                continue
            if ti == tj and not (ei or ej):
                continue
            permissible += 1.0
            pi, pj = predicted[i], predicted[j]
            if ti != tj:
                short, long_ = (i, j) if ti < tj else (j, i)
                ps, pl = predicted[short], predicted[long_]
                if ps > pl:
                    concordance += 1.0
                elif ps == pl:
                    concordance += 0.5
            elif ei and ej:
                concordance += 1.0 if pi == pj else 0.5
            else:
                dead = i if ei else j
                other = j if ei else i
                if predicted[dead] > predicted[other]:
                    concordance += 1.0
                else:
                    concordance += 0.5
    if permissible == 0.0:
        raise ValueError("survrsf: no permissible pairs -- every pair "
                         "has its shorter time censored")
    return {"c_index": concordance / permissible,
            "concordance": concordance, "permissible": permissible,
            "prediction_error": 1.0 - concordance / permissible}


def vimp(fit, X, variables=None, seed=1):
    r"""Sec. 7: random daughter assignment at splits on :math:`x`."""
    base = mortality(fit, X, oob=True, seed=seed)
    base_pe = c_index(fit["times"], fit["events"],
                      base)["prediction_error"]
    variables = (range(len(X[0])) if variables is None
                 else [int(v) for v in variables])
    out = {}
    for v in variables:
        m = mortality(fit, X, oob=True, random_variable=v, seed=seed)
        pe = c_index(fit["times"], fit["events"],
                     m)["prediction_error"]
        out[v] = pe - base_pe
    return RichResult(payload={
        "estimate": max(out.values()) if out else 0.0,
        "vimp": out, "baseline_error": base_pe,
        "note": "VIMP is the change in error for a fresh case if x "
                "were unavailable, NOT the change from regrowing the "
                "forest without x",
        "method": "variable importance by random daughter "
                  "assignment; Ishwaran et al. (2008) Sec. 7",
    })


def cheatsheet():
    return ("survrsf: bootstrap survival trees split on the log-rank "
            "statistic, Nelson-Aalen in each terminal node, averaged "
            "into an out-of-bag ensemble CHF. Conservation of events "
            "(Lemma 1) is exact: the hazard summed over ALL observed "
            "times, censored included, equals the number of deaths. "
            "Mortality is that sum, a count of deaths, not a "
            "probability. Error is 1 - C with Harrell's four-step "
            "recipe. Two of the paper's four splitting rules are "
            "implemented; the other two are refused with their "
            "citations rather than guessed.")


# compact alias per ledger/NAMING.md
random_survival_forest = forest
