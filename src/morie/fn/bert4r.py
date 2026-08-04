# morie.fn -- slice s03 (rootcoder007/morie)
"""BERT4Rec: the masked-item objective and its evaluation.

Source consulted (FETCHED): Sun, F. et al. (2019).  BERT4Rec: sequential
recommendation with bidirectional encoder representations from
transformer.  *CIKM* 28, 1441-1450 (arXiv:1904.06690).  The training
objective is the cloze task: a proportion rho of the items in each user
sequence is replaced by [mask], and the loss is

    L = (1 / |S^u_m|) sum_(v_m in S^u_m) -log P( v_m = v*_m | S'_u )

i.e. the negative log likelihood of the true item at each masked
position, averaged over the masked positions.  At test time exactly one
[mask] is appended to the end of the sequence, which is the paper's
device for turning a bidirectional model into a next-item recommender.

DETERMINISM.  Which positions are masked is not drawn: positions are
selected on a fixed stride so that the masked fraction is rho and the
selection reproduces in both arms.  The paper's random masking is a
training device; the loss it defines is what is computed here.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["bert4rec"]

_EPS = 1e-300


def bert4rec(seqs, K=10, scores=None, rho=0.2):
    """Cloze loss over masked positions, plus ranking metrics at K.

    Parameters
    ----------
    seqs : 2-D array-like
        User sequences of item ids (zero-based), one row per user.
    K : int
        Cut-off for HR@K and NDCG@K.
    scores : 3-D array-like, optional
        Model scores over the item vocabulary at each masked position:
        ``scores[u][j]`` is the score vector for the j-th masked position
        of user u.  When absent a uniform model is assumed, which makes
        the loss log V exactly -- the honest null baseline.
    rho : float
        Masking proportion.

    Returns
    -------
    RichResult with payload:
        estimate : the cloze loss
        loss     : same as estimate
        hr       : hit rate at K
        ndcg     : NDCG at K
        n_masked : number of masked positions
    """
    S = k.mat(seqs)
    V = 0
    for row in S:
        for it in row:
            if int(it) + 1 > V:
                V = int(it) + 1
    total = 0.0
    nm = 0
    hits = 0.0
    ndcg = 0.0
    for u in range(len(S)):
        row = S[u]
        L = len(row)
        step = int(1.0 / float(rho)) if rho > 0.0 else L
        if step < 1:
            step = 1
        pos = list(range(step - 1, L, step))
        for j in range(len(pos)):
            target = int(row[pos[j]])
            if scores is None:
                p = 1.0 / V if V else 0.0
                rank = (V + 1) / 2.0
            else:
                sc = k.vec(scores[u][j])
                pr = k.softmax(sc)
                p = pr[target]
                rank = 1.0
                for c in range(len(sc)):
                    if sc[c] > sc[target]:
                        rank += 1.0
            total -= math.log(p if p > _EPS else _EPS)
            nm += 1
            if rank <= float(K):
                hits += 1.0
                ndcg += 1.0 / math.log(rank + 1.0, 2.0)
    return RichResult(
        title="BERT4Rec cloze objective",
        summary_lines=[("masked", nm)],
        payload={
            "estimate": total / nm if nm else float("nan"),
            "loss": total / nm if nm else float("nan"),
            "hr": hits / nm if nm else float("nan"),
            "ndcg": ndcg / nm if nm else float("nan"),
            "n_masked": nm,
            "n_items": V,
            "method": "BERT4Rec cloze loss with HR@K and NDCG@K (Sun et al. 2019)",
        },
    )


def cheatsheet():
    return "bert4r: BERT4Rec"
