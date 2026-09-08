# morie.fn -- function file (rootcoder007/morie)
r"""GloVe -- global vectors for word representation.

Pennington, Socher & Manning (2014). The model is a weighted least
squares regression on the logarithm of the word-word co-occurrence
counts. Their eq. (8):

.. math:: J = \sum_{i,j=1}^{V} f(X_{ij})
          \left(w_i^{\top}\tilde w_j + b_i + \tilde b_j
                - \log X_{ij}\right)^2

with the weighting function of eq. (9),

.. math:: f(x) = \begin{cases}
          (x / x_{\max})^{\alpha} & x < x_{\max}\\
          1 & \text{otherwise,}\end{cases}

and their stated defaults :math:`x_{\max} = 100` and
:math:`\alpha = 3/4` ("we fix to x_max = 100 for all our experiments";
"alpha = 3/4 gives a modest improvement over a linear version with
alpha = 1"). The stub returned ``mean(corpus)``.

Three details that are easy to get wrong and are in the paper:

**Zero counts are skipped, not weighted to zero.** f(0) = 0 is
property 1, but log 0 is undefined, so the sum runs over nonzero
:math:`X_{ij}` only. Evaluating the term first and multiplying by zero
afterwards produces a NaN that then poisons every gradient.

**Two sets of vectors, and the sum is what you use.** The paper keeps
:math:`w` and :math:`\tilde w`, notes they are equivalent up to
initialisation, and recommends :math:`w + \tilde w` as the final
representation -- which also acts as a variance reduction. The default
here is that sum, with the separate matrices returned as well.

**The context window is harmonic.** Sec. 4.2: a token d positions away
contributes 1/d to the count, "so that very distant word pairs are
expected to contain less relevant information". A flat window is a
different model and gives different vectors.

Training is AdaGrad, as in the paper, because the per-parameter step
size matters here: word frequencies span orders of magnitude and a
single global learning rate either crawls on rare words or diverges on
frequent ones.

References
----------
Pennington, J., Socher, R. & Manning, C. D. (2014) "GloVe: Global
Vectors for Word Representation", *Proceedings of the 2014 Conference
on Empirical Methods in Natural Language Processing (EMNLP)*,
1532-1543, doi:10.3115/v1/D14-1162. Equations (8) and (9), Sec. 4.2
for the harmonic weighting and the AdaGrad training.

Duchi, J., Hazan, E. & Singer, Y. (2011) "Adaptive subgradient methods
for online learning and stochastic optimization", *Journal of Machine
Learning Research* 12, 2121-2159 -- the AdaGrad the paper uses.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["glove", "cooccurrence", "glove_weight", "glove_loss"]


def glove_weight(x, x_max=100.0, alpha=0.75):
    """Eq. (9). f(0) = 0, non-decreasing, capped at 1."""
    x = float(x)
    if x <= 0.0:
        return 0.0
    if x >= float(x_max):
        return 1.0
    return (x / float(x_max)) ** float(alpha)


def cooccurrence(corpus, window=10, harmonic=True, min_count=1):
    """Word-word co-occurrence counts over a symmetric context window.

    Sec. 4.2: "we use a decreasing weighting function, so that word
    pairs that are d words apart contribute 1/d to the total count".
    `harmonic=False` gives the flat window for comparison; it is a
    different model, not a speed-up.
    """
    docs = _as_docs(corpus)
    counts = {}
    for doc in docs:
        for t in doc:
            counts[t] = counts.get(t, 0) + 1
    vocab = sorted(t for t, c in counts.items() if c >= int(min_count))
    index = {t: i for i, t in enumerate(vocab)}
    X = {}
    w = int(window)
    if w < 1:
        raise ValueError("cooccurrence: window must be at least 1, got %r"
                         % (window,))
    for doc in docs:
        ids = [index[t] for t in doc if t in index]
        for pos, i in enumerate(ids):
            lo = max(0, pos - w)
            for other in range(lo, pos):
                j = ids[other]
                d = pos - other
                inc = 1.0 / d if harmonic else 1.0
                X[(i, j)] = X.get((i, j), 0.0) + inc
                X[(j, i)] = X.get((j, i), 0.0) + inc
    return X, vocab, index


def glove_loss(X, W, Wt, b, bt, x_max=100.0, alpha=0.75):
    """Eq. (8), evaluated over the nonzero entries only."""
    total = 0.0
    for (i, j), x in X.items():
        if x <= 0.0:
            continue
        pred = sum(W[i][d] * Wt[j][d] for d in range(len(W[i]))) \
            + b[i] + bt[j]
        diff = pred - math.log(x)
        total += glove_weight(x, x_max, alpha) * diff * diff
    return total


def glove(corpus, dim=50, window=10, epochs=25, lr=0.05, x_max=100.0,
          alpha=0.75, harmonic=True, min_count=1, seed=0,
          combine="sum"):
    r"""Fit GloVe vectors.

    Parameters
    ----------
    corpus : sequence
        A list of documents, each a list of tokens, or a list of
        strings which are split on whitespace.
    dim : int
        Vector dimension.
    combine : {"sum", "w", "wtilde", "concat"}
        Which representation to return. The paper recommends the sum
        of the two vector sets.

    Returns
    -------
    RichResult
        ``estimate`` is the vector matrix, with ``vocab`` giving the
        row order. ``loss_history`` is eq. (8) evaluated at the end of
        each epoch -- recomputable from the returned parameters, and
        checked to be so. ``running_loss`` is the SGD running total
        accumulated during the epoch, which is the cheaper progress
        signal but is not the objective at any single parameter value.

    Examples
    --------
    Words that co-occur end up closer than words that do not::

        r = glove([["a", "b", "a", "b"], ["c", "d", "c", "d"]], dim=8)
        r["vocab"], r["estimate"]
    """
    if combine not in ("sum", "w", "wtilde", "concat"):
        raise ValueError("glove: combine must be 'sum', 'w', 'wtilde' or "
                         "'concat', got %r" % (combine,))
    d = int(dim)
    if d < 1:
        raise ValueError("glove: dim must be at least 1, got %r" % (dim,))
    X, vocab, index = cooccurrence(corpus, window=window,
                                   harmonic=harmonic,
                                   min_count=min_count)
    V = len(vocab)
    if V < 2:
        raise ValueError(
            "glove: the corpus has %d word(s) above min_count=%r; GloVe "
            "factorises a co-occurrence matrix and needs at least two"
            % (V, min_count))
    if not X:
        raise ValueError("glove: no co-occurrences within the window, so "
                         "eq. (8) has no terms")

    rng = np.random.default_rng(int(seed))
    scale = 0.5 / d
    W = [[(float(rng.uniform()) - 0.5) * scale for _ in range(d)]
         for _ in range(V)]
    Wt = [[(float(rng.uniform()) - 0.5) * scale for _ in range(d)]
          for _ in range(V)]
    b = [(float(rng.uniform()) - 0.5) * scale for _ in range(V)]
    bt = [(float(rng.uniform()) - 0.5) * scale for _ in range(V)]
    # AdaGrad accumulators, initialised to 1 as in the reference code
    gW = [[1.0] * d for _ in range(V)]
    gWt = [[1.0] * d for _ in range(V)]
    gb = [1.0] * V
    gbt = [1.0] * V

    entries = sorted(X.items())
    history = []          # eq. (8) at the end of each epoch
    running = []          # the SGD running total, for comparison
    eta = float(lr)
    for _ in range(int(epochs)):
        total = 0.0
        for (i, j), x in entries:
            if x <= 0.0:
                continue
            wi, wj = W[i], Wt[j]
            pred = sum(wi[t] * wj[t] for t in range(d)) + b[i] + bt[j]
            diff = pred - math.log(x)
            fw = glove_weight(x, x_max, alpha)
            total += fw * diff * diff
            g = 2.0 * fw * diff          # d/d(pred) of the term
            for t in range(d):
                gi = g * wj[t]
                gj = g * wi[t]
                W[i][t] -= eta * gi / math.sqrt(gW[i][t])
                Wt[j][t] -= eta * gj / math.sqrt(gWt[j][t])
                gW[i][t] += gi * gi
                gWt[j][t] += gj * gj
            b[i] -= eta * g / math.sqrt(gb[i])
            bt[j] -= eta * g / math.sqrt(gbt[j])
            gb[i] += g * g
            gbt[j] += g * g
        running.append(total)
        # eq. (8) evaluated at the parameters this epoch ended with,
        # rather than the running sum accumulated while they moved
        history.append(glove_loss(X, W, Wt, b, bt, x_max, alpha))

    if combine == "sum":
        vecs = [[W[i][t] + Wt[i][t] for t in range(d)] for i in range(V)]
    elif combine == "w":
        vecs = [list(r) for r in W]
    elif combine == "wtilde":
        vecs = [list(r) for r in Wt]
    else:
        vecs = [list(W[i]) + list(Wt[i]) for i in range(V)]

    return RichResult(payload={
        "estimate": vecs,
        "vectors": vecs,
        "vocab": vocab,
        "index": index,
        "W": W, "W_tilde": Wt, "b": b, "b_tilde": bt,
        "cooccurrence": X,
        "loss_history": history,
        "running_loss": running,
        "final_loss": history[-1] if history else float("nan"),
        "n_vocab": V, "n_pairs": len(entries), "dim": d,
        "window": int(window), "harmonic": bool(harmonic),
        "x_max": float(x_max), "alpha": float(alpha),
        "combine": combine,
        "method": "GloVe weighted least squares on log co-occurrence, "
                  "Pennington, Socher & Manning (2014) eqs. (8)-(9)",
    })


def _as_docs(corpus):
    if corpus is None:
        raise ValueError("glove: corpus must not be None")
    docs = []
    for item in corpus:
        if isinstance(item, str):
            docs.append(item.split())
        else:
            docs.append([str(t) for t in item])
    if not docs:
        raise ValueError("glove: the corpus is empty")
    return docs


def cheatsheet():
    return ("glove: J = sum f(X_ij)(w_i.wt_j + b_i + bt_j - log X_ij)^2 "
            "with f(x) = (x/xmax)^alpha capped at 1, xmax=100, "
            "alpha=3/4 (Pennington-Socher-Manning 2014 eqs.8-9). "
            "Harmonic 1/d context window; AdaGrad; final vector w + wt.")
