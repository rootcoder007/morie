"""K-class gradient tree boosting (ESL Alg. 10.4)."""

import math

from ._richresult import RichResult
from .esldct import _best_split

__all__ = ["esl_gbm_multiclass"]


def esl_gbm_multiclass(X, g, M=100, nu=0.1, max_depth=1, min_leaf=1, query=None, leaf_scale=None):
    r"""Multinomial deviance boosting with regression trees (ESL Alg. 10.4).

    With :math:`p_k(x) = e^{f_k(x)}/\sum_\ell e^{f_\ell(x)}` (eq 10.21), each of the M
    rounds fits, for every class k, a least-squares regression tree to the
    negative gradient :math:`r_{ik} = y_{ik} - p_k(x_i)`, then replaces each leaf
    value by the one-step Newton estimate of eq 10.57,
    :math:`\gamma_{jkm} = \frac{K-1}{K}\frac{\sum_{x_i\in R_{jkm}} r_{ik}}{\sum_{x_i\in R_{jkm}}|r_{ik}|(1-|r_{ik}|)}`,
    and updates :math:`f_k \leftarrow f_k + \nu\sum_j\gamma_{jkm}I(x\in R_{jkm})`
    (eq 10.58's symmetric parametrisation). The functions start at zero.

    Parameters
    ----------
    X : N x p nested sequence
    g : sequence of N labels
    M : int
        Rounds.
    nu : float
        Shrinkage, in (0, 1].
    max_depth, min_leaf
        Tree controls (depth 1 = stumps).
    query : m x p nested sequence, optional
    leaf_scale : float, optional
        Factor on the Newton leaf step; default (K - 1)/K as in ESL and Friedman
        (2001). ``gbm``'s multinomial distribution uses 1.

    Returns
    -------
    RichResult
        ``prob`` (query x K), ``prediction``, ``deviance_path`` (training
        multinomial deviance after each round), ``classes``, ``trees``.

    References
    ----------
    Friedman, J. (2001). Greedy function approximation: a gradient boosting
    machine. Annals of Statistics 29, 1189-1232.
    """
    from . import _array_core as np

    rows = [[float(v) for v in r] for r in X]
    labels = list(g)
    classes = sorted(set(labels), key=repr)
    N, K = len(rows), len(classes)
    if len(labels) != N or K < 2 or not 0 < nu <= 1:
        raise ValueError("need matching X and g, at least two classes and 0 < nu <= 1")
    Xa = np.asarray(rows)
    Y = [[1.0 if lab == c else 0.0 for c in classes] for lab in labels]
    F = [[0.0] * K for _ in range(N)]

    def grow(idx, r, depth):
        yy = np.asarray([r[i] for i in idx])
        if depth >= max_depth or len(idx) < 2 * min_leaf or max(r[i] for i in idx) == min(r[i] for i in idx):
            return {"leaf": True, "idx": idx}
        best = _best_split(Xa[idx], yy, min_leaf)
        if best is None or best[3] <= 0:
            return {"leaf": True, "idx": idx}
        _, j, thr, _ = best
        left = [i for i in idx if rows[i][j] <= thr]
        right = [i for i in idx if rows[i][j] > thr]
        return {
            "leaf": False,
            "feature": int(j),
            "threshold": float(thr),
            "left": grow(left, r, depth + 1),
            "right": grow(right, r, depth + 1),
        }

    def leafify(node, r):
        if node["leaf"]:
            num = sum(r[i] for i in node["idx"])
            den = sum(abs(r[i]) * (1 - abs(r[i])) for i in node["idx"])
            return {"leaf": True, "value": sc * num / den if den > 0 else 0.0}
        return {**node, "left": leafify(node["left"], r), "right": leafify(node["right"], r)}

    def predict_tree(node, x):
        while not node["leaf"]:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node["value"]

    def softmax(f):
        m = max(f)
        e = [math.exp(v - m) for v in f]
        s = sum(e)
        return [v / s for v in e]

    sc = (K - 1) / K if leaf_scale is None else float(leaf_scale)
    trees, dev = [], []
    for _ in range(int(M)):
        P = [softmax(f) for f in F]
        round_trees = []
        for k in range(K):
            r = [Y[i][k] - P[i][k] for i in range(N)]
            t = leafify(grow(list(range(N)), r, 0), r)
            round_trees.append(t)
        for i in range(N):
            for k in range(K):
                F[i][k] += nu * predict_tree(round_trees[k], rows[i])
        trees.append(round_trees)
        dev.append(-2 * sum(math.log(softmax(F[i])[classes.index(labels[i])]) for i in range(N)))
    Q = rows if query is None else [[float(v) for v in r] for r in query]
    probs = []
    for x in Q:
        f = [nu * sum(predict_tree(rt[k], x) for rt in trees) for k in range(K)]
        probs.append(softmax(f))
    return RichResult(
        title="K-class gradient boosting",
        summary_lines=[("rounds", len(trees))],
        payload={
            "prob": probs,
            "prediction": [classes[max(range(K), key=lambda k: (pp[k], -k))] for pp in probs],
            "deviance_path": dev,
            "classes": classes,
            "trees": trees,
        },
    )


def cheatsheet():
    return "eslgbk: per class a tree on y_ik - p_ik, leaves (K-1)/K sum r / sum |r|(1-|r|) (ESL 10.57), shrunk by nu"
