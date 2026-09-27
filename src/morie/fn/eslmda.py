"""Mixture discriminant analysis by EM (ESL sec 12.7)."""

import math

from ._richresult import RichResult
from .nlsgn import _inverse

__all__ = ["esl_mda"]


def _chol_logdet(S):
    n = len(S)
    L = [[0.0] * n for _ in range(n)]
    for a in range(n):
        for b in range(a + 1):
            s = S[a][b] - sum(L[a][k] * L[b][k] for k in range(b))
            L[a][b] = math.sqrt(s) if a == b else s / L[b][b]
    return 2 * sum(math.log(L[a][a]) for a in range(n))


def esl_mda(X, g, subclasses=2, weights=None, query=None, max_iter=1000, tol=1e-12):
    r"""Gaussian mixtures within classes with a common covariance, fitted by EM.

    ESL eqs 12.59-12.61: class k has density
    :math:`\sum_r\pi_{kr}\phi(x;\mu_{kr},\Sigma)` and
    :math:`\Pr(G=k|x) \propto \Pi_k\sum_r\pi_{kr}\phi(x;\mu_{kr},\Sigma)` (eq 12.60).
    The E-step computes the within-class subclass responsibilities
    :math:`W(c_{kr}|x_i,g_i)`; the M-step is a weighted LDA: subclass means and
    mixing proportions, and the pooled covariance
    :math:`\Sigma = \sum_{k,r}\sum_{g_i=k}W_{ikr}(x_i-\mu_{kr})(x_i-\mu_{kr})^T/N`.

    Parameters
    ----------
    X : N x p nested sequence
    g : sequence of N labels
    subclasses : int or sequence
        Subclasses per class.
    weights : dict {class: n_k x R_k list}, optional
        Starting responsibilities; by default each class is split into
        contiguous blocks after sorting on its first coordinate.
    query : M x p nested sequence, optional
    max_iter, tol
        EM controls (change in the log-likelihood).

    Returns
    -------
    RichResult
        ``posterior`` (query x classes), ``prediction``, ``means``,
        ``mixing``, ``covariance``, ``loglik``, ``iterations``, ``classes``.

    References
    ----------
    Hastie, T. & Tibshirani, R. (1996). Discriminant analysis by Gaussian
    mixtures. JRSS B 58, 155-176.
    """
    rows = [[float(v) for v in r] for r in X]
    labels = list(g)
    classes = sorted(set(labels), key=repr)
    N, p, K = len(rows), len(rows[0]), len(classes)
    R = [int(subclasses)] * K if isinstance(subclasses, int) else [int(v) for v in subclasses]
    idx = {c: [i for i in range(N) if labels[i] == c] for c in classes}
    if weights is None:
        W = {}
        for k, c in enumerate(classes):
            order = sorted(idx[c], key=lambda i: (rows[i][0], i))
            nk = len(order)
            W[c] = {
                i: [1.0 if r == min(pos * R[k] // nk, R[k] - 1) else 0.0 for r in range(R[k])]
                for pos, i in enumerate(order)
            }
    else:
        W = {c: {i: [float(v) for v in weights[c][a]] for a, i in enumerate(idx[c])} for c in classes}
    prior = [len(idx[c]) / N for c in classes]

    def mstep(W):
        mu, pi = {}, {}
        S = [[0.0] * p for _ in range(p)]
        for k, c in enumerate(classes):
            mu[c], pi[c] = [], []
            for r in range(R[k]):
                w = [W[c][i][r] for i in idx[c]]
                sw = sum(w)
                m = [sum(wi * rows[i][j] for wi, i in zip(w, idx[c])) / sw for j in range(p)]
                mu[c].append(m)
                pi[c].append(sw / len(idx[c]))
                for wi, i in zip(w, idx[c]):
                    d = [rows[i][j] - m[j] for j in range(p)]
                    for a in range(p):
                        for b in range(p):
                            S[a][b] += wi * d[a] * d[b] / N
        return mu, pi, S

    def logdens(x, mu, Si, ld):
        d = [x[j] - mu[j] for j in range(p)]
        return -0.5 * (p * math.log(2 * math.pi) + ld + sum(d[a] * Si[a][b] * d[b] for a in range(p) for b in range(p)))

    prev, it, conv = -math.inf, 0, False
    for _ in range(max_iter):
        it += 1
        mu, pi, S = mstep(W)
        Si, ld = _inverse(S), _chol_logdet(S)
        ll = 0.0
        for k, c in enumerate(classes):
            for i in idx[c]:
                lp = [
                    math.log(pi[c][r]) + logdens(rows[i], mu[c][r], Si, ld) if pi[c][r] > 0 else -math.inf
                    for r in range(R[k])
                ]
                m = max(lp)
                e = [math.exp(v - m) for v in lp]
                s = sum(e)
                W[c][i] = [v / s for v in e]
                ll += m + math.log(s)
        if abs(ll - prev) < tol * (1 + abs(ll)):
            conv = True
            break
        prev = ll
    mu, pi, S = mstep(W)
    Si, ld = _inverse(S), _chol_logdet(S)
    Q = rows if query is None else [[float(v) for v in r] for r in query]
    post = []
    for x in Q:
        lc = []
        for k, c in enumerate(classes):
            lp = [math.log(pi[c][r]) + logdens(x, mu[c][r], Si, ld) for r in range(R[k]) if pi[c][r] > 0]
            m = max(lp)
            lc.append(math.log(prior[k]) + m + math.log(sum(math.exp(v - m) for v in lp)))
        m = max(lc)
        e = [math.exp(v - m) for v in lc]
        post.append([v / sum(e) for v in e])
    return RichResult(
        title="Mixture discriminant analysis",
        summary_lines=[("loglik", ll), ("iterations", it)],
        payload={
            "posterior": post,
            "prediction": [classes[max(range(K), key=lambda k: (pp[k], -k))] for pp in post],
            "means": {c: mu[c] for c in classes},
            "mixing": {c: pi[c] for c in classes},
            "covariance": S,
            "loglik": ll,
            "iterations": it,
            "converged": conv,
            "classes": classes,
        },
    )


def cheatsheet():
    return "eslmda: EM over subclass responsibilities within classes, common Sigma; posterior by ESL 12.60"
