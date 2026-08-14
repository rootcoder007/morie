# morie.fn -- function file (rootcoder007/morie)
r"""GRU4Rec: session-based recommendation with a ranking loss.

Most recommenders assume a user profile. Many real systems have none
-- the user is anonymous and all that exists is the current session's
clicks. The task is then to predict the next event from the sequence
so far, which is a sequence problem, but with two properties that stop
an off-the-shelf RNN working.

**Sessions have wildly different lengths, so mini-batches cannot be
built the usual way.** Some sessions have two events, others hundreds,
and padding or fragmenting both destroy what is being modelled -- how
a session *evolves*. The answer is **session-parallel mini-batches**:
the first events of :math:`X` sessions form the first batch, their
second events the next, and when a session ends the next available one
takes its slot with the corresponding hidden state reset. Sessions are
assumed independent, which is what makes the reset correct.

**The item set is huge and only the top matters, so the loss must
rank.** Cross-entropy over tens of thousands of items pushes the
target's score up without pushing negatives down, and the paper
reports it was numerically stable in only 10 of 100 random runs. Two
pairwise losses work instead:

.. math:: L_{BPR} = -\frac{1}{N_S}\sum_j
          \log\sigma(\hat r_{s,i} - \hat r_{s,j}),

.. math:: L_{TOP1} = \frac{1}{N_S}\sum_j \Big[
          \sigma(\hat r_{s,j} - \hat r_{s,i}) + \sigma(\hat
          r_{s,j}^2)\Big].

TOP1 is the relative rank of the target with the indicator smoothed by
a sigmoid, **plus a regularisation term**. That second term is not
decoration: without it, positive items that also serve as negatives
push all scores upward without bound. The term forces negative scores
toward zero and is deliberately in the same range as the rank term.
Both losses are implemented and the anchor shows the runaway that the
regulariser prevents.

References
----------
Hidasi, B., Karatzoglou, A., Baltrunas, L. & Tikk, D. (2016)
"Session-based Recommendations with Recurrent Neural Networks",
*International Conference on Learning Representations (ICLR 2016)*,
arXiv:1511.06939. Sec. 1 and 3 (the absence of a user profile in many
real systems and the need for a ranking loss). Sec. 3.1 (session-
parallel mini-batches: why fragmenting or padding sessions of
2 to a few hundred events would misrepresent how a session evolves,
and the hidden-state reset when a session is replaced). Sec. 3.1.3
(the BPR and TOP1 ranking losses, with TOP1's regularisation term
added because positive items acting as negatives drive scores
upward). Sec. 4 (cross-entropy was numerically stable in only 10 of
100 runs, while both pairwise losses performed well; a single GRU
layer was best).

Rendle, S., Freudenthaler, C., Gantner, Z. & Schmidt-Thieme, L.
(2009) "BPR: Bayesian Personalized Ranking from Implicit Feedback",
*UAI 2009*, 452-461, arXiv:1205.2618. The BPR loss reused here;
implemented in :mod:`bprMF`.

Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares,
F., Schwenk, H. & Bengio, Y. (2014) "Learning Phrase Representations
using RNN Encoder-Decoder for Statistical Machine Translation",
*EMNLP 2014*, 1724-1734, doi:10.3115/v1/D14-1179. The GRU unit.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["session_parallel_batches", "bpr_loss", "top1_loss",
           "gru_step", "recall_at_k", "mrr_at_k"]

_EPS = 1e-12
_LOSSES = ("bpr", "top1", "cross_entropy")


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0


def session_parallel_batches(sessions, batch_size):
    r"""Sec. 3.1: parallel slots, with a reset flag when one is
    refilled.

    Returns a list of steps; each step gives the input item, the
    target item, and which slots were reset -- the reset is what makes
    treating sessions as independent correct.
    """
    S = [[int(v) for v in s] for s in sessions]
    if any(len(s) < 2 for s in S):
        raise ValueError("gru4r: every session needs at least 2 "
                         "events")
    B = int(batch_size)
    if B < 1 or B > len(S):
        raise ValueError("gru4r: batch_size must lie in 1..%d, got %d"
                         % (len(S), B))
    slot = list(range(B))
    pos = [0] * B
    nxt = B
    steps = []
    while True:
        x, y, reset, alive = [], [], [], False
        for b in range(B):
            if slot[b] is None:
                x.append(None)
                y.append(None)
                reset.append(False)
                continue
            s = S[slot[b]]
            if pos[b] + 1 >= len(s):
                if nxt < len(S):
                    slot[b], pos[b] = nxt, 0
                    nxt += 1
                    reset.append(True)
                    s = S[slot[b]]
                else:
                    slot[b] = None
                    x.append(None)
                    y.append(None)
                    reset.append(False)
                    continue
            else:
                reset.append(False)
            x.append(s[pos[b]])
            y.append(s[pos[b] + 1])
            pos[b] += 1
            alive = True
        if not alive:
            break
        steps.append({"input": x, "target": y, "reset": reset})
    return {"steps": steps, "n_steps": len(steps), "batch_size": B,
            "n_sessions": len(S),
            "note": "a slot's hidden state is reset when a new "
                    "session takes it, because sessions are assumed "
                    "independent"}


def bpr_loss(r_target, r_negatives):
    r""":math:`-\frac{1}{N_S}\sum_j \log\sigma(\hat r_i - \hat r_j)`."""
    neg = [float(v) for v in k.vec(r_negatives)]
    if not neg:
        raise ValueError("gru4r: at least one negative is needed")
    return -sum(math.log(max(_sig(float(r_target) - v), _EPS))
                for v in neg) / len(neg)


def top1_loss(r_target, r_negatives, regularize=True):
    r"""TOP1. ``regularize=False`` drops the
    :math:`\sigma(\hat r_j^2)` term whose absence lets scores run
    away."""
    neg = [float(v) for v in k.vec(r_negatives)]
    if not neg:
        raise ValueError("gru4r: at least one negative is needed")
    rank = sum(_sig(v - float(r_target)) for v in neg) / len(neg)
    if not regularize:
        return rank
    return rank + sum(_sig(v * v) for v in neg) / len(neg)


def gru_step(x, h, Wz, Uz, Wr, Ur, Wh, Uh):
    r"""One GRU update, for a single layer as the paper found best."""
    n = len(h)

    def lin(W, U, xv, hv):
        return [sum(W[o][j] * xv[j] for j in range(len(xv)))
                + sum(U[o][j] * hv[j] for j in range(len(hv)))
                for o in range(n)]

    z = [_sig(v) for v in lin(Wz, Uz, x, h)]
    r = [_sig(v) for v in lin(Wr, Ur, x, h)]
    hh = [math.tanh(v) for v in
          lin(Wh, Uh, x, [r[i] * h[i] for i in range(n)])]
    return [(1.0 - z[i]) * h[i] + z[i] * hh[i] for i in range(n)]


def recall_at_k(ranked, target, kk=20):
    r"""Whether the target appears in the top :math:`k`."""
    return 1.0 if int(target) in list(ranked)[:int(kk)] else 0.0


def mrr_at_k(ranked, target, kk=20):
    r"""Reciprocal rank, zero beyond :math:`k`."""
    top = list(ranked)[:int(kk)]
    return 1.0 / (top.index(int(target)) + 1.0) \
        if int(target) in top else 0.0


def cheatsheet():
    return ("gru4r: no user profile, just the current session. "
            "SESSION-PARALLEL mini-batches -- slot b holds one "
            "session, refilled with a hidden-state RESET when it ends "
            "-- because padding or fragmenting destroys how a session "
            "evolves. Cross-entropy over a huge item set was stable in "
            "only 10 of 100 runs; use BPR or TOP1. TOP1 = smoothed "
            "relative rank PLUS sigma(r_neg^2), and that second term "
            "is load-bearing: without it positives acting as negatives "
            "drive every score upward.")


# compact alias per ledger/NAMING.md
gruforrecommendation = session_parallel_batches

# public names resolved by fn/_lazy_map.json
gru4rec = session_parallel_batches
