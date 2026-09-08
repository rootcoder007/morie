# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""LDA by collapsed Gibbs sampling (Blei et al. 2003; Griffiths and
Steyvers 2004; Alammar Ch 5)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["alammar_lda_topic_distribution"]


def alammar_lda_topic_distribution(documents, n_topics, alpha=0.1,
                                   beta=0.01, n_iter=200, seed=1):
    """Collapsed Gibbs: z ~ P(z = k | rest) proportional to
    (n_dk + alpha)(n_kw + beta)/(n_k + V beta), driven by the shared
    LCG so both languages walk the same chain.

    Returns theta (documents x topics) and phi (topics x vocab), both
    rows summing to 1 -- asserted in the tests, not assumed.

    References: Alammar and Grootendorst, Ch 5; Griffiths and Steyvers
    (2004), Eq 5.
    """
    docs = [[str(w) for w in d] for d in documents]
    K = int(n_topics)
    if K < 2:
        raise ValueError("n_topics must be at least 2.")
    if not docs or any(len(d) == 0 for d in docs):
        raise ValueError("every document must contain at least one token.")
    a = float(alpha); b = float(beta)
    if a <= 0 or b <= 0:
        raise ValueError("alpha and beta must be positive.")
    vocab = sorted({w for d in docs for w in d})
    V = len(vocab)
    widx = {w: i for i, w in enumerate(vocab)}
    s = int(seed) % 2 ** 32

    def unif():
        nonlocal s
        s = (1664525 * s + 1013904223) % 2 ** 32
        return (s + 0.5) / 2 ** 32

    n_dk = np.zeros((len(docs), K))
    n_kw = np.zeros((K, V))
    n_k = np.zeros(K)
    z = []
    for di, d in enumerate(docs):
        zs = []
        for w in d:
            k = int(unif() * K)
            zs.append(k)
            n_dk[di, k] += 1
            n_kw[k, widx[w]] += 1
            n_k[k] += 1
        z.append(zs)
    for _ in range(int(n_iter)):
        for di, d in enumerate(docs):
            for wi, w in enumerate(d):
                k = z[di][wi]
                v = widx[w]
                n_dk[di, k] -= 1; n_kw[k, v] -= 1; n_k[k] -= 1
                p = (n_dk[di] + a) * (n_kw[:, v] + b) / (n_k + V * b)
                p = p / p.sum()
                u = unif()
                k = int(np.searchsorted(np.cumsum(p), u, side="right"))
                k = min(k, K - 1)
                z[di][wi] = k
                n_dk[di, k] += 1; n_kw[k, v] += 1; n_k[k] += 1
    theta = (n_dk + a) / (n_dk.sum(axis=1, keepdims=True) + K * a)
    phi = (n_kw + b) / (n_kw.sum(axis=1, keepdims=True) + V * b)
    return RichResult(payload={
        "theta": [[float(v) for v in r] for r in theta],
        "phi": [[float(v) for v in r] for r in phi],
        "vocabulary": vocab,
        "estimate": float(theta[0, 0]), "n": len(docs),
        "method": "LDA collapsed Gibbs (Griffiths and Steyvers 2004)"})


def cheatsheet():
    return "alldat: collapsed Gibbs on the shared LCG; theta and phi rows sum to 1"
