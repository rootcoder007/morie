r"""Membership inference against machine learning models (shadow training).

Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017) "Membership
Inference Attacks Against Machine Learning Models", *IEEE Symposium on
Security and Privacy*, 3-18.

Given black-box access to a target model -- a record goes in, a vector of
class probabilities comes out -- decide whether that record was in the
model's training set. The attack "exploits the observation that machine
learning models often behave differently on the data that they were trained
on versus the data that they 'see' for the first time", and the paper's
contribution is a way to *learn* that difference without ever seeing the
target's training data.

**Shadow training.** The attacker trains :math:`k` shadow models on data
drawn like the target's, and for each one knows the ground truth. Querying
shadow model :math:`i` with its own training records gives outputs labelled
``"in"``; querying it with a disjoint test set gives outputs labelled
``"out"``. Those labelled prediction vectors are the training set for the
attack model (figure 3). Because "the attack model is a collection of models,
one for each output class of the target model", a separate binary classifier
is fitted per true class, which is what makes the attack sensitive to the
class-conditional shape of the output vector rather than to overall
confidence alone.

**Where shadow data comes from.** All three of the paper's routes are here
(``synthesis``):

* ``"model"`` -- synthesis from the target model itself, Algorithm 1,
  implemented as printed: hill-climb by proposing a record that changes
  :math:`k` random features of the last accepted one, accept only if
  :math:`y_c` increases, halve :math:`k` (floored at :math:`k_{min}`) after
  :math:`\mathit{rej}_{max}` consecutive rejections, and once
  :math:`y_c > \mathit{conf}_{min}` and :math:`c = \arg\max y`, return the
  record with probability :math:`y_c`; give up after
  :math:`\mathit{iter}_{max}` iterations;
* ``"marginals"`` -- "independently sampling the value of each feature from
  its own marginal distribution", which the paper found "very effective";
* ``"noisy"`` -- real data with a fraction of features perturbed, the
  paper's 10-20% flips.

**Metrics** are the paper's: precision, "what fraction of records inferred as
members are indeed members of the training dataset", and recall, "what
fraction of the training dataset's members are correctly inferred as
members".

The classifier used for the target, the shadows and the attack models is
supplied by the caller as ``train_fn(X, y) -> predict_fn``; a native
multinomial logistic regression (:func:`logistic_trainer`) is the default so
the attack is runnable without one. The attack is only as strong as the gap
it feeds on -- on a target that does not overfit there is nothing to find,
which the paper says outright and this module's anchors demonstrate.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["memb", "membership_inference", "logistic_trainer",
           "knn_trainer",
           "attack_dataset", "synthesize", "synthesize_marginals",
           "synthesize_noisy", "precision_recall"]


def _rng(seed):
    return np.random.default_rng(seed)


def logistic_trainer(l2=1e-3, epochs=300, lr=0.5, seed=0):
    r"""A native multinomial logistic regression, returned as a
    ``train_fn``.

    Softmax over :math:`W x + b`, full-batch gradient ascent on the
    penalised log-likelihood. It is here so the attack can be run without
    any external learner; pass your own ``train_fn`` for anything serious.
    ``l2`` is what controls how much the model overfits, and therefore how
    much there is for the attack to find.
    """
    def train(X, y):
        n = len(X)
        if n == 0:
            raise ValueError("memb: cannot train on an empty dataset")
        d = len(X[0])
        classes = sorted(set(y))
        idx = dict((c, k) for k, c in enumerate(classes))
        C = len(classes)
        W = [[0.0] * d for _ in range(C)]
        b = [0.0] * C
        for _ in range(int(epochs)):
            gW = [[0.0] * d for _ in range(C)]
            gb = [0.0] * C
            for i in range(n):
                z = [sum(W[k][j] * X[i][j] for j in range(d)) + b[k]
                     for k in range(C)]
                mx = max(z)
                ez = [math.exp(v - mx) for v in z]
                ssum = sum(ez)
                pr = [v / ssum for v in ez]
                t = idx[y[i]]
                for k in range(C):
                    err = (1.0 if k == t else 0.0) - pr[k]
                    gb[k] += err
                    for j in range(d):
                        gW[k][j] += err * X[i][j]
            for k in range(C):
                b[k] += lr * gb[k] / n
                for j in range(d):
                    W[k][j] += lr * (gW[k][j] / n - l2 * W[k][j])

        def predict(rows):
            out = []
            for x in rows:
                z = [sum(W[k][j] * x[j] for j in range(d)) + b[k]
                     for k in range(C)]
                mx = max(z)
                ez = [math.exp(v - mx) for v in z]
                ssum = sum(ez)
                out.append([v / ssum for v in ez])
            return out
        predict.classes = classes
        return predict
    return train


def knn_trainer(k=1, smoothing=1e-3):
    r"""A native k-nearest-neighbour learner, returned as a ``train_fn``.

    Included because the attack needs something that actually memorises:
    with :math:`k = 1` every training record is returned with near-certainty
    at its own label while unseen records are decided by whichever neighbour
    happens to be closest, which is the train/test gap the attack reads.
    ``smoothing`` keeps the output vector strictly positive.
    """
    k = int(k)
    if k < 1:
        raise ValueError("memb: k must be >= 1")

    def train(X, y):
        if not X:
            raise ValueError("memb: cannot train on an empty dataset")
        classes = sorted(set(y))
        idx = dict((c, t) for t, c in enumerate(classes))
        rows = [list(v) for v in X]
        labs = list(y)

        def predict(query):
            out = []
            for q in query:
                d = sorted(range(len(rows)),
                           key=lambda i: sum((a - b) ** 2
                                             for a, b in zip(rows[i], q)))
                votes = [smoothing] * len(classes)
                for i in d[:k]:
                    votes[idx[labs[i]]] += 1.0
                tot = sum(votes)
                out.append([v / tot for v in votes])
            return out
        predict.classes = classes
        return predict
    return train


def attack_dataset(model_predict, in_X, in_y, out_X, out_y):
    """Figure 3: label a shadow model's outputs ``in`` and ``out``.

    Returns ``(features, labels, classes)``: the prediction vector for each
    record, 1 for a member of that model's training set and 0 otherwise, and
    the true class of each record (the attack keeps one model per class).
    """
    rows, lab, cls = [], [], []
    for X, y, flag in ((in_X, in_y, 1), (out_X, out_y, 0)):
        if not X:
            continue
        for vec, c in zip(model_predict(X), y):
            rows.append(list(vec))
            lab.append(flag)
            cls.append(c)
    return rows, lab, cls


def synthesize(target_predict, c, n_features, feature_values=None,
               k_max=None, k_min=1, conf_min=0.8, iter_max=1000,
               rej_max=10, seed=0):
    r"""Algorithm 1: synthesise a record the target classifies as ``c``.

    Implemented as printed. ``feature_values`` gives the allowed values per
    feature (binary by default, so a proposal flips them); ``k`` starts at
    ``k_max``, halves -- :math:`\lceil k/2 \rceil` -- after ``rej_max``
    consecutive rejections, and never falls below ``k_min``.

    Returns the record, or ``None`` for the algorithm's :math:`\bot`
    ("failed to synthesize").
    """
    n_features = int(n_features)
    if n_features < 1:
        raise ValueError("memb: n_features must be >= 1")
    if not 0.0 < conf_min < 1.0:
        raise ValueError("memb: conf_min must lie in (0, 1)")
    if k_min < 1:
        raise ValueError("memb: k_min must be >= 1")
    k_max = n_features if k_max is None else int(k_max)
    if k_max < k_min:
        raise ValueError("memb: k_max must be at least k_min")
    rng = _rng(seed)
    vals = feature_values or [[0.0, 1.0]] * n_features
    if len(vals) != n_features:
        raise ValueError("memb: feature_values must have one entry per "
                         "feature")

    def rand_record(base=None, k=None):
        if base is None:
            return [vals[j][int(rng.random() * len(vals[j]))]
                    for j in range(n_features)]
        x = list(base)
        picks = set()
        while len(picks) < min(k, n_features):
            picks.add(int(rng.random() * n_features))
        for j in picks:
            choices = [v for v in vals[j] if v != x[j]] or vals[j]
            x[j] = choices[int(rng.random() * len(choices))]
        return x

    x = rand_record()
    y_best = 0.0
    x_best = list(x)
    j = 0
    k = k_max
    for _ in range(int(iter_max)):
        y = list(target_predict([x])[0])
        if c >= len(y):
            raise ValueError("memb: class %r is outside the target's output "
                             "vector" % (c,))
        yc = y[c]
        if yc >= y_best:
            if yc > conf_min and c == max(range(len(y)), key=lambda t: y[t]):
                if rng.random() < yc:
                    return x
            x_best = list(x)
            y_best = yc
            j = 0
        else:
            j += 1
            if j > rej_max:
                k = max(k_min, -(-k // 2))
                j = 0
        x = rand_record(x_best, k)
    return None


def synthesize_marginals(X, n, seed=0):
    """Statistics-based synthesis: each feature drawn from its own
    marginal, independently of the others."""
    if not X:
        raise ValueError("memb: no data to take marginals from")
    rng = _rng(seed)
    d = len(X[0])
    cols = [[row[j] for row in X] for j in range(d)]
    out = []
    for _ in range(int(n)):
        out.append([cols[j][int(rng.random() * len(cols[j]))]
                    for j in range(d)])
    return out


def synthesize_noisy(X, fraction=0.1, feature_values=None, seed=0):
    """Noisy real data: resample a fraction of the features of each record,
    the paper's 10-20% flips for binary features."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("memb: fraction must lie in [0, 1]")
    rng = _rng(seed)
    d = len(X[0])
    vals = feature_values or [sorted(set(row[j] for row in X))
                              for j in range(d)]
    out = []
    for row in X:
        x = list(row)
        for j in range(d):
            if rng.random() < fraction:
                choices = [v for v in vals[j] if v != x[j]] or vals[j]
                x[j] = choices[int(rng.random() * len(choices))]
        out.append(x)
    return out


def precision_recall(pred, truth):
    r"""The paper's metrics.

    Precision is :math:`tp/(tp+fp)`, "what fraction of records inferred as
    members are indeed members"; recall is :math:`tp/(tp+fn)`, "what
    fraction of the training dataset's members are correctly inferred".
    """
    tp = sum(1 for p, t in zip(pred, truth) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(pred, truth) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(pred, truth) if p == 0 and t == 1)
    tn = sum(1 for p, t in zip(pred, truth) if p == 0 and t == 0)
    n = tp + fp + fn + tn
    return {"precision": tp / float(tp + fp) if tp + fp else float("nan"),
            "recall": tp / float(tp + fn) if tp + fn else float("nan"),
            "accuracy": (tp + tn) / float(n) if n else float("nan"),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _sorted_features(vec, top=None):
    s = sorted(vec, reverse=True)
    return s if top is None else s[:top]


def memb(target_predict, shadow_data, eval_in, eval_out, train_fn=None,
         attack_train_fn=None, n_shadow=None, sort_features=False,
         threshold=0.5):
    r"""Run a shadow-training membership inference attack.

    Parameters
    ----------
    target_predict : callable
        Black-box access: ``target_predict(rows) -> list of probability
        vectors``. Nothing else about the target is used.
    shadow_data : sequence of ``(train_X, train_y, test_X, test_y)``
        One tuple per shadow model: the data it is trained on and a disjoint
        set it never sees. Generate these with :func:`synthesize`,
        :func:`synthesize_marginals` or :func:`synthesize_noisy`.
    eval_in, eval_out : ``(X, y)``
        Records known to be in and out of the *target's* training set, used
        to score the attack. These are the evaluation ground truth, never
        given to the attack.
    train_fn : callable, optional
        ``train_fn(X, y) -> predict_fn`` for the shadow models. Defaults to
        :func:`logistic_trainer`. It should match the target's learner as
        closely as the attacker can manage -- "the shadow models must be
        trained in a similar way to the target model".
    attack_train_fn : callable, optional
        Learner for the per-class attack models. Defaults to
        :func:`logistic_trainer`.
    n_shadow : int, optional
        Use only the first this many shadow specifications. "The more
        shadow models, the more accurate the attack model will be."
    sort_features : bool
        Feed the attack model the prediction vector sorted descending
        instead of in class order. Useful when the class count is large or
        the attacker cannot align classes; off by default because the
        paper's attack keeps the vector as it is, one model per class.
    threshold : float
        Membership probability above which a record is called a member.

    Returns
    -------
    RichResult
        ``estimate`` / ``metrics`` are precision, recall and accuracy on the
        evaluation records; ``predictions`` and ``scores`` are per record;
        ``per_class`` breaks the metrics down by true class; ``n_shadow``,
        ``attack_train_size`` describe the attack itself.

    Examples
    --------
    With shadow sets already in hand::

        res = memb(target.predict, shadows, (in_X, in_y), (out_X, out_y))
        res["metrics"]["precision"], res["metrics"]["recall"]

    References
    ----------
    Shokri, Stronati, Song & Shmatikov (2017) *IEEE S&P*, sections IV-V,
    Algorithm 1 and figures 1-3.
    """
    if train_fn is None:
        train_fn = logistic_trainer()
    if attack_train_fn is None:
        attack_train_fn = logistic_trainer()
    specs = list(shadow_data)
    if n_shadow is not None:
        specs = specs[:int(n_shadow)]
    if not specs:
        raise ValueError("memb: at least one shadow model is needed")

    rows, labels, classes = [], [], []
    for spec in specs:
        tr_X, tr_y, te_X, te_y = spec
        if not tr_X:
            raise ValueError("memb: a shadow model has no training data")
        shadow = train_fn(tr_X, tr_y)
        r, lb, cl = attack_dataset(shadow, tr_X, tr_y, te_X, te_y)
        rows.extend(r)
        labels.extend(lb)
        classes.extend(cl)
    if not rows:
        raise ValueError("memb: the shadow models produced no attack data")

    # one attack model per output class of the target (section IV-A)
    per_class = {}
    for c in sorted(set(classes)):
        idx = [t for t in range(len(rows)) if classes[t] == c]
        if len(set(labels[t] for t in idx)) < 2:
            continue
        feats = [_sorted_features(rows[t]) if sort_features else rows[t]
                 for t in idx]
        per_class[c] = attack_train_fn(feats, [labels[t] for t in idx])

    if not per_class:
        raise ValueError("memb: no class had both in and out examples, so no "
                         "attack model could be trained")

    eval_X = list(eval_in[0]) + list(eval_out[0])
    eval_y = list(eval_in[1]) + list(eval_out[1])
    truth = [1] * len(eval_in[0]) + [0] * len(eval_out[0])
    outputs = target_predict(eval_X) if eval_X else []
    scores, preds = [], []
    for vec, c in zip(outputs, eval_y):
        model = per_class.get(c)
        if model is None:
            scores.append(float("nan"))
            preds.append(0)
            continue
        feat = _sorted_features(list(vec)) if sort_features else list(vec)
        pr = model([feat])[0]
        member = pr[1] if len(pr) > 1 else pr[0]
        scores.append(member)
        preds.append(1 if member >= threshold else 0)

    metrics = precision_recall(preds, truth)
    by_class = {}
    for c in sorted(set(eval_y)):
        sel = [t for t in range(len(eval_y)) if eval_y[t] == c]
        if sel:
            by_class[c] = precision_recall([preds[t] for t in sel],
                                           [truth[t] for t in sel])
    return RichResult(payload={
        "estimate": metrics,
        "metrics": metrics,
        "per_class": by_class,
        "predictions": preds,
        "scores": scores,
        "truth": truth,
        "n_shadow": len(specs),
        "attack_train_size": len(rows),
        "attack_classes": sorted(per_class),
        "threshold": float(threshold),
        "note": "the attack can only find a gap that exists: against a "
                "target that does not overfit, precision falls to the "
                "base rate (Shokri et al. 2017, section VII)",
        "method": "shadow-trained membership inference (Shokri et al. 2017)",
    })


def cheatsheet():
    return ("memb: membership inference (Shokri et al. 2017). Black-box "
            "output vector in, member/non-member out. Train k SHADOW "
            "models on data distributed like the target's, where you DO "
            "know membership; their outputs on their own training data are "
            "labelled 'in' and on a disjoint test set 'out'; that labelled "
            "set trains the attack model -- one per output class, since "
            "the tell is class-conditional. Shadow data from Algorithm 1 "
            "synthesis against the target, from feature marginals, or from "
            "noisy real data. Metrics are precision and recall over "
            "members. The attack lives on the train/test gap: no "
            "overfitting, no attack.")


# compact alias per ledger/NAMING.md
membership_inference = memb
