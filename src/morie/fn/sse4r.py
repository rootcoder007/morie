# morie.fn -- function file (rootcoder007/morie)
r"""SSE-PT: a personalised Transformer, regularised by shared
embeddings.

A self-attentive sequential recommender models what a user did, in
order, and ignores *who* the user is. That is a deliberate
simplification with a cost: two users with the same recent items get
the same recommendation, and everything the model knows about their
differing long-run taste is thrown away.

**Personalisation is a concatenation, not an extra tower.** The user
embedding is concatenated to **every item embedding** in the sequence
before attention, so the user modulates every position rather than
being added once at the end. ``personalise`` does exactly that, and
the anchor checks that two different users with an *identical* history
receive different outputs -- which fails immediately if the user
embedding is dropped.

**But personalisation is also where the parameters explode.** One
embedding per user, with each user contributing only a handful of
sequences, is a recipe for memorisation.

**Stochastic Shared Embeddings is the regulariser that pays for it.**
During training, an embedding is replaced at random -- with probability
:math:`p` -- by *another* embedding from the same table. Not zeroed,
which is dropout; **exchanged**, which forces the representations of
different users (or items) to remain mutually compatible rather than
each memorising its own rows. At :math:`p = 0` it is the identity,
which the anchor asserts exactly, and the realised replacement rate
must track :math:`p`.

References
----------
Wu, L., Li, S., Hsieh, C.-J. & Sharpnack, J. (2020) "SSE-PT:
Sequential Recommendation Via Personalized Transformer", *Proceedings
of the 14th ACM Conference on Recommender Systems (RecSys '20)*,
328-337, doi:10.1145/3383313.3412258; earlier circulated as "Temporal
Collaborative Ranking Via Personalized Transformer",
arXiv:1908.05435. The observation that existing self-attentive
sequential models are not personalised and that different users' rating
patterns are treated alike; the personalised Transformer concatenating
a user embedding with each item embedding in the input sequence; and
the use of Stochastic Shared Embeddings regularisation, replacing an
embedding by another from the same table with a given probability
during training, to make the large per-user embedding table trainable.

Wu, L., Li, S., Hsieh, C.-J. & Sharpnack, J. (2019) "Stochastic
Shared Embeddings: Data-driven Regularization of Embedding Layers",
*NeurIPS 2019*, arXiv:1905.10630. The SSE regulariser itself.

Kang, W.-C. & McAuley, J. (2018) "Self-Attentive Sequential
Recommendation", *ICDM 2018*, 197-206, arXiv:1808.09781. SASRec, the
unpersonalised model being extended; implemented in :mod:`sasRec`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["personalise", "sse_replace", "expected_replacement",
           "parameter_count", "predict_next"]

_EPS = 1e-12


def personalise(item_embeddings, user_embedding):
    r"""Concatenate the user vector to EVERY item in the sequence.

    Appending it once at the end would let attention ignore it; here
    it modulates every position.
    """
    I = [[float(v) for v in r] for r in k.mat(item_embeddings)]
    u = [float(v) for v in k.vec(user_embedding)]
    if not I:
        raise ValueError("sse4r: the sequence is empty")
    return {"sequence": [list(row) + list(u) for row in I],
            "item_dim": len(I[0]), "user_dim": len(u),
            "width": len(I[0]) + len(u), "length": len(I),
            "note": "every position carries the user, so two users "
                    "with the same history diverge"}


def sse_replace(indices, table_size, p=0.0, seed=0):
    r"""Replace an index by ANOTHER index with probability :math:`p`.

    Not zeroing -- that is dropout. Exchanging forces different rows to
    stay mutually compatible instead of each memorising its own.
    """
    idx = [int(v) for v in indices]
    n = int(table_size)
    pr = float(p)
    if n < 1:
        raise ValueError("sse4r: the embedding table is empty")
    if not 0.0 <= pr <= 1.0:
        raise ValueError("sse4r: p must lie in [0,1], got %r" % (p,))
    if any(v < 0 or v >= n for v in idx):
        raise ValueError("sse4r: an index is outside the table")
    if pr == 0.0:
        return {"indices": list(idx), "replaced": [], "p": 0.0,
                "rate": 0.0,
                "note": "p = 0 is exactly the identity"}
    rng = np.random.default_rng(seed)
    out, rep = [], []
    for i, v in enumerate(idx):
        if float(rng.uniform()) < pr:
            j = int(float(rng.uniform()) * n) % n
            out.append(j)
            if j != v:
                rep.append((i, v, j))
        else:
            out.append(v)
    return {"indices": out, "replaced": rep, "p": pr,
            "rate": sum(1 for a in range(len(idx))
                        if out[a] != idx[a]) / float(len(idx)),
            "note": "the replacement is drawn from the SAME table"}


def expected_replacement(p, table_size):
    r"""The realised rate is :math:`p(1-1/n)`, not :math:`p`.

    A replacement drawn uniformly sometimes lands on the original
    index, and the anchor checks against this closed form rather than
    against :math:`p`.
    """
    pr, n = float(p), int(table_size)
    if n < 1:
        raise ValueError("sse4r: the table is empty")
    return {"expected_rate": pr * (1.0 - 1.0 / n), "p": pr,
            "table_size": n,
            "note": "self-replacement is invisible, so the observed "
                    "rate is below p"}


def parameter_count(n_users, n_items, user_dim, item_dim):
    r"""Where the personalisation cost lands."""
    nu, ni = int(n_users), int(n_items)
    du, di = int(user_dim), int(item_dim)
    if min(nu, ni, du, di) < 1:
        raise ValueError("sse4r: every count must be positive")
    up, ip = nu * du, ni * di
    return {"user_params": up, "item_params": ip,
            "total": up + ip,
            "user_share": up / float(up + ip),
            "note": "one row per user with few sequences each -- which "
                    "is why SSE is needed rather than optional"}


def predict_next(sequence, user_embedding, item_table, attend=None,
                 top_k=3):
    r"""Score every item for this user and this history.

    The user term in the final dot product is the same for every
    candidate, so it cannot change a ranking on its own: the
    personalisation has to act through the ATTENTION over the
    personalised sequence, which is what the default does.
    """
    seq = personalise(sequence, user_embedding)["sequence"]
    u = [float(v) for v in k.vec(user_embedding)]
    T = [[float(v) for v in r] for r in k.mat(item_table)]
    if attend is None:
        # Attention with the PERSONALISED last position as the query.
        # A plain mean would pool the user term into a constant that
        # cancels out of every score, so the model would be
        # unpersonalised again -- the attention is where the user
        # actually acts.
        d = len(seq[0])
        di = d - len(u)
        qy = seq[-1]
        # item-item similarity PLUS a user-item interaction. The
        # interaction has to be user x item: scoring the user against
        # itself contributes u'u to every position alike, so a user
        # vector and its negation would be indistinguishable and the
        # model would be unpersonalised in disguise.
        m = min(len(u), di)
        sc = [(sum(qy[a] * seq[t][a] for a in range(di))
               + sum(u[a] * seq[t][a] for a in range(m)))
              / math.sqrt(d) for t in range(len(seq))]
        m = max(sc)
        e = [math.exp(v - m) for v in sc]
        z = sum(e)
        w = [v / z for v in e]
        ctx = [sum(w[t] * seq[t][a] for t in range(len(seq)))
               for a in range(d)]
    else:
        ctx = [float(v) for v in k.vec(attend(seq))]
    di = len(ctx) - len(u)
    scores = []
    for row in T:
        if len(row) != di:
            raise ValueError("sse4r: the item table is %d-wide but "
                             "the item part of the context is %d"
                             % (len(row), di))
        scores.append(sum(ctx[a] * row[a] for a in range(di))
                      + sum(ctx[di + a] * u[a] for a in range(len(u))))
    order = sorted(range(len(scores)), key=lambda j: -scores[j])
    kk = min(int(top_k), len(order))
    return RichResult(payload={
        "estimate": order[:kk], "top_k": order[:kk],
        "scores": scores, "context": ctx,
        "method": "personalised Transformer recommendation; Wu, Li, "
                  "Hsieh & Sharpnack (2020)",
        "note": "the user term is present at every position, so the "
                "same history gives different users different "
                "answers",
    })


def cheatsheet():
    return ("sse4r: a self-attentive sequential recommender models WHAT "
            "was clicked and ignores WHO clicked, so two users with the "
            "same recent items get the same answer. Fix it by "
            "CONCATENATING a user embedding to EVERY item in the "
            "sequence -- appended once at the end, attention could "
            "ignore it. That adds one row per user, each with few "
            "sequences, so it memorises; SSE regularises by REPLACING "
            "an embedding with another from the SAME table with "
            "probability p. Not zeroing (that is dropout) -- exchanging, "
            "which keeps different rows mutually compatible. The "
            "observed rate is p(1-1/n), not p.")


# compact alias per ledger/NAMING.md
ssept = predict_next

# public names resolved by fn/_lazy_map.json
ssepta_seq = predict_next
sseptaseq = predict_next
