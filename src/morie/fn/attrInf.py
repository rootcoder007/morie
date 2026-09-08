# morie.fn -- function file (rootcoder007/morie)
r"""Model inversion: recovering a sensitive feature from a released model.

Fredrikson, M., Jha, S., & Ristenpart, T. (2015) "Model Inversion
Attacks that Exploit Confidence Information and Basic Countermeasures",
*CCS '15*, 1322-1333. doi:10.1145/2810103.2813677

The attacker holds a model :math:`f`, a target's non-sensitive features
:math:`x_2, \dots, x_t`, that target's label :math:`y`, marginal priors
:math:`p_i` for each feature, and an error model. The question is the
value of the *sensitive* feature :math:`x_1`.

**The generic estimator** (the paper's Figure 2) is a maximum a
posteriori estimate. For each candidate value :math:`v` it scores

.. math::

   r_v \;=\; \mathrm{err}\bigl(y, f(v, x_2, \dots, x_t)\bigr)
             \cdot \prod_i p_i(x_i),

and returns :math:`\arg\max_v r_v`. It is "the least-biased maximum a
posteriori estimate for :math:`x_1` given the available information",
which is exactly why it is the right baseline: it minimises the
attacker's misprediction rate under what the attacker knows.

**Against a decision tree** the error model is a confusion matrix rather
than a Gaussian, :math:`\mathrm{err}(y, y') \propto \Pr[f(x) = y' \mid y
\text{ is the true label}]`, which is the *black-box* attack. The paper
reports it has "a prohibitively high false positive rate", and the
anchor here measures that rather than taking it on faith.

**The white-box-with-counts estimator** is the paper's improvement. A
tree is :math:`f(x) = \sum_i w_i \varphi_i(x)` with one basis function
per root-to-leaf path; white-box access also gives :math:`n_i`, the
number of training rows down path :math:`i`, so :math:`p_i = n_i / N`
estimates how often the joint prior traverses that path. Equation 1:

.. math::

   \Pr[x_1 = v \mid (s_1 \vee \dots \vee s_m) \wedge x_K = v_K]
   \;\propto\;
   \frac{1}{\sum_j p_j \varphi_j(v)}
   \sum_{1 \le i \le m} p_i \varphi_i(v) \cdot \Pr[x_1 = v].

The counts carry information about the *joint* distribution over
features that the per-feature marginals cannot, which is where the
improvement comes from. With more than one unknown feature the
estimator sums Equation 1 over the unknowns, as the paper states.

Trees are given as nested ``{"feature": i, "branches": {value: subtree}}``
dicts with leaves ``{"label": y, "count": n}``; ``count`` is only needed
for the white-box route.
"""

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult

__all__ = [
    "attrInf",
    "attribute_inference",
    "map_invert",
    "wbwc_invert",
    "tree_predict",
    "tree_paths",
    "confusion_error",
]

_MODES = ("blackbox", "whitebox")


def _leaf(node):
    return isinstance(node, dict) and "label" in node


def tree_predict(tree, x):
    """The label of the leaf ``x`` reaches."""
    node = tree
    while not _leaf(node):
        try:
            i, br = node["feature"], node["branches"]
        except (KeyError, TypeError):
            raise ValueError("attrInf: a node needs 'feature' and "
                             "'branches', or 'label' for a leaf")
        try:
            v = x[i]
        except (IndexError, KeyError):
            raise ValueError("attrInf: no value supplied for feature %r, "
                             "which the tree needs" % (i,))
        if v is None:
            raise ValueError("attrInf: no value supplied for feature %r, "
                             "which the tree needs" % (i,))
        if v not in br:
            raise ValueError("attrInf: no branch for value %r of feature "
                             "%r" % (v, i))
        node = br[v]
    return node["label"]


def tree_paths(tree):
    r"""Every root-to-leaf path: the basis functions :math:`\varphi_i`.

    Each is ``{"constraints": {feature: value}, "label": y,
    "count": n}`` -- the path is active exactly when every constraint
    holds.
    """
    out = []

    def walk(node, cons):
        if _leaf(node):
            out.append({"constraints": dict(cons), "label": node["label"],
                        "count": float(node.get("count", 0.0))})
            return
        i, br = node["feature"], node["branches"]
        for v, child in br.items():
            cons[i] = v
            walk(child, cons)
            del cons[i]

    walk(tree, {})
    if not out:
        raise ValueError("attrInf: the tree has no paths")
    return out


def confusion_error(C, labels=None):
    r"""``err(y, y') = Pr[f(x) = y' | y true]``, row-normalised.

    ``C[i][j]`` counts training rows whose true label is ``i`` and whose
    predicted label is ``j``.
    """
    rows = [[float(v) for v in r] for r in C]
    if not rows or any(len(r) != len(rows) for r in rows):
        raise ValueError("attrInf: the confusion matrix must be square")
    if any(v < 0 for r in rows for v in r):
        raise ValueError("attrInf: confusion counts cannot be negative")
    labels = list(labels) if labels is not None else list(range(len(rows)))
    if len(labels) != len(rows):
        raise ValueError("attrInf: one label per confusion-matrix row")
    idx = dict((b, k) for k, b in enumerate(labels))
    table = {}
    for i, y in enumerate(labels):
        tot = sum(rows[i])
        for j, yp in enumerate(labels):
            table[(y, yp)] = (rows[i][j] / tot) if tot > 0 else 0.0

    def err(y, yp):
        if y not in idx or yp not in idx:
            raise ValueError("attrInf: unknown label in the error model")
        return table[(y, yp)]

    return err


def map_invert(model, y, known, candidates, err, priors, sensitive=0):
    r"""Figure 2: the generic MAP inversion attack.

    ``known`` maps feature index to value for everything the attacker
    knows; ``priors`` maps feature index to ``{value: probability}``.
    Returns the scores and the arg max.
    """
    if not candidates:
        raise ValueError("attrInf: no candidate values to score")
    scores = {}
    for v in candidates:
        x = dict(known)
        x[sensitive] = v
        n = max(list(x) + [sensitive]) + 1
        vec = [x.get(i) for i in range(n)]
        e = err(y, model(vec))
        prod = 1.0
        for i, val in x.items():
            p = priors.get(i)
            if p is not None:
                prod *= p.get(val, 0.0)
        scores[v] = e * prod
    best = max(sorted(scores, key=lambda v: str(v)),
               key=lambda v: scores[v])
    return {"scores": scores, "estimate": best}


def wbwc_invert(tree, known, candidates, priors, sensitive=0,
                unknown=None):
    r"""Equation 1: the white-box-with-counts estimator.

    ``unknown`` lists any further features the attacker does not know;
    Equation 1 is summed over them, as the paper specifies.
    """
    if not candidates:
        raise ValueError("attrInf: no candidate values to score")
    paths = tree_paths(tree)
    N = sum(p["count"] for p in paths)
    if N <= 0:
        raise ValueError("attrInf: white-box inversion needs path counts; "
                         "give each leaf a 'count'")
    unknown = list(unknown or [])
    scores = {}
    for v in candidates:
        assign = dict(known)
        assign[sensitive] = v
        active, total = 0.0, 0.0
        for p in paths:
            pi = p["count"] / N
            # a path is compatible if no constraint contradicts what is
            # known; constraints on unknown features are summed over,
            # which is what "summing (1) over the unknown variables" is
            ok = True
            for i, val in p["constraints"].items():
                if i in unknown:
                    continue
                if i in assign and assign[i] != val:
                    ok = False
                    break
            if ok:
                active += pi
            total += pi
        prior = priors.get(sensitive, {}).get(v, 0.0)
        denom = active if active > 0 else 1.0
        scores[v] = (active / denom) * active * prior if active > 0 \
            else 0.0
    best = max(sorted(scores, key=lambda v: str(v)),
               key=lambda v: scores[v])
    return {"scores": scores, "estimate": best, "n_paths": len(paths),
            "N": N}


def attrInf(tree, targets, priors, confusion=None, labels=None,
            sensitive=0, mode="blackbox", candidates=None, unknown=None):
    """Run the attack over a set of targets and score it.

    Each target is ``{"known": {...}, "label": y}`` plus, when the truth
    is known and the attack is being evaluated, ``"truth"``.
    """
    if mode not in _MODES:
        raise ValueError("attrInf: mode must be one of %s" % (_MODES,))
    paths = tree_paths(tree)
    if candidates is None:
        cand = sorted(set(c[sensitive] for p in paths
                          for c in [p["constraints"]]
                          if sensitive in c), key=lambda v: str(v))
        if not cand:
            cand = sorted(priors.get(sensitive, {}),
                          key=lambda v: str(v))
    else:
        cand = list(candidates)
    if not cand:
        raise ValueError("attrInf: no candidate values for the sensitive "
                         "feature")
    err = None
    if mode == "blackbox":
        if confusion is None:
            raise ValueError("attrInf: the black-box attack needs a "
                             "confusion matrix")
        err = confusion_error(confusion, labels)

    guesses, correct, n_truth = [], 0, 0
    tp = fp = fn = 0
    positive = cand[-1]
    for t in targets:
        known = dict(t.get("known", {}))
        y = t.get("label")
        if mode == "blackbox":
            got = map_invert(lambda x: tree_predict(tree, x), y, known,
                             cand, err, priors, sensitive)
        else:
            got = wbwc_invert(tree, known, cand, priors, sensitive,
                              unknown)
        guesses.append(got["estimate"])
        if "truth" in t:
            n_truth += 1
            if got["estimate"] == t["truth"]:
                correct += 1
            if got["estimate"] == positive and t["truth"] == positive:
                tp += 1
            elif got["estimate"] == positive:
                fp += 1
            elif t["truth"] == positive:
                fn += 1
    acc = (correct / float(n_truth)) if n_truth else None
    prec = (tp / float(tp + fp)) if (tp + fp) else None
    rec = (tp / float(tp + fn)) if (tp + fn) else None
    return RichResult(payload={
        "estimate": guesses,
        "guesses": guesses,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "false_positives": fp,
        "true_positives": tp,
        "mode": mode,
        "candidates": cand,
        "n_paths": len(paths),
        "n_targets": len(targets),
        "method": ("model inversion (Fredrikson, Jha & Ristenpart 2015): "
                   "%s MAP estimate of the sensitive feature"
                   % ("generic Figure 2" if mode == "blackbox"
                      else "white-box-with-counts, Equation 1")),
        "note": ("the black-box route uses err(y, y') from the confusion "
                 "matrix and the paper reports it has a prohibitively "
                 "high false positive rate; the white-box route adds the "
                 "per-path training counts, which carry joint-"
                 "distribution information the marginals cannot"),
    })


attribute_inference = attrInf


def cheatsheet():
    return ("attrInf: model inversion (Fredrikson, Jha & Ristenpart "
            "2015). Figure 2 scores each candidate value v of the "
            "sensitive feature by err(y, f(v, x_2..x_t)) times the "
            "product of the marginal priors and takes the arg max -- the "
            "least-biased MAP estimate. Against a tree the error model "
            "is the confusion matrix (black box); the white-box-with-"
            "counts estimator of eq.1 instead weights each root-to-leaf "
            "path by n_i/N, which carries joint information the "
            "marginals do not.")
