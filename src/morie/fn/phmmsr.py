# morie.fn -- function file (rootcoder007/morie)
r"""HMMER3: making profile HMM search as fast as BLAST.

Profile hidden Markov models are more sensitive than pairwise
comparison, and were far more expensive -- which is why heuristic
tools were used instead. HMMER3's contribution is an acceleration
*pipeline* that keeps the sensitivity.

**The MSV filter, and why its statistics matter.** The "multiple
segment Viterbi" algorithm computes an optimal **sum of multiple
ungapped local alignment segments**, using a striped vector-parallel
layout. Dropping gaps is what makes it vectorisable. The crucial
property is not the speed but the distribution: **MSV scores follow
the same statistical distribution as gapped optimal local alignment
scores** -- a Gumbel -- so a p-value can be computed for an MSV score
directly, and the filter's threshold is a *statistical* one rather
than an arbitrary cutoff. The anchor checks the Gumbel tail rather
than trusting the claim.

**Sparse rescaling** gives a further 20-fold acceleration of the
Forward/Backward algorithms. Probabilities underflow over a long
sequence; rescaling at every position is the textbook fix and costs a
division per cell. Rescaling only when the values actually approach
the floor keeps the numbers safe at a fraction of the cost, and
``sparse_rescale`` reports how often it fired.

**A pipeline, not a single algorithm.** High-scoring MSV hits are
passed on for reanalysis with the full Forward/Backward model. The
benchmark claim is that the filter sacrifices *negligible*
sensitivity, which is the only thing that would justify it -- a fast
filter that loses true positives is not an acceleration, it is a
different, worse method.

References
----------
Eddy, S. R. (2011) "Accelerated Profile HMM Searches", *PLoS
Computational Biology* 7(10), e1002195,
doi:10.1371/journal.pcbi.1002195. The MSV algorithm computing an
optimal sum of multiple ungapped local alignment segments by a striped
vector-parallel approach; MSV scores following the same statistical
distribution as gapped optimal local alignment scores, allowing rapid
evaluation of significance and use as a heuristic filter; the 20-fold
acceleration of Forward/Backward by sparse rescaling; the pipeline in
which high-scoring MSV hits are reanalysed with the full HMM; and the
benchmarks showing negligible sensitivity sacrificed, with HMMER3
100-1000 fold faster than HMMER2 and about as fast as BLAST for
protein searches.

Farrar, M. (2007) "Striped Smith-Waterman speeds database searches six
times over other SIMD implementations", *Bioinformatics* 23(2),
156-161, doi:10.1093/bioinformatics/btl582. The striped layout reused.

Durbin, R., Eddy, S. R., Krogh, A. & Mitchison, G. (1998)
*Biological Sequence Analysis*, Cambridge University Press,
doi:10.1017/CBO9780511790492. Profile HMMs.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["msv_score", "gumbel_pvalue", "sparse_rescale",
           "search_pipeline", "striped_layout"]

_EPS = 1e-12


def striped_layout(length, vector_width=4):
    r"""The striped index order that makes the recursion vectorisable.

    Positions are interleaved so a vector's lanes have no
    within-vector dependency.
    """
    L, w = int(length), int(vector_width)
    if L < 1 or w < 1:
        raise ValueError("phmmsr: the length and width must be "
                         "positive")
    q = (L + w - 1) // w
    order = []
    for i in range(q):
        for j in range(w):
            p = j * q + i
            if p < L:
                order.append(p)
    return {"order": order, "segments": q, "width": w,
            "note": "lanes are independent within a vector, which is "
                    "what permits the parallel update"}


def msv_score(seq, profile, tau=0.02, lam=0.7):
    r"""Optimal sum of multiple UNGAPPED local segments.

    No gaps, which is what makes it vectorisable, and it still scores
    on the same scale as the gapped alignment it filters for.
    """
    s = list(seq)
    P = [[float(v) for v in r] for r in k.mat(profile)]
    M = len(P)
    if M < 1:
        raise ValueError("phmmsr: the profile is empty")
    best = 0.0
    xmx = 0.0
    dp = [0.0] * M
    for i in range(len(s)):
        prev = list(dp)
        for j in range(M):
            emit = P[j][int(s[i])] if isinstance(s[i], int) \
                else P[j][0]
            src = xmx + math.log(max(float(tau), _EPS)) \
                if j == 0 else prev[j - 1]
            dp[j] = max(src + emit, 0.0)
        xmx = max(xmx, max(dp))
        best = max(best, xmx)
    return {"score": best,
            "note": "ungapped segments only, summed -- the gap "
                    "recursion is what could not be vectorised"}


def gumbel_pvalue(score, mu, lam):
    r"""The Gumbel tail: :math:`P = 1 - \exp(-e^{-\lambda(S-\mu)})`.

    MSV scores share this distribution with gapped local alignment
    scores, which is what turns the filter threshold into a p-value.
    """
    l = float(lam)
    if l <= 0.0:
        raise ValueError("phmmsr: lambda must be positive")
    z = -l * (float(score) - float(mu))
    return 1.0 - math.exp(-math.exp(z)) if z < 700 else 1.0


def sparse_rescale(values, floor=1e-30, target=1.0):
    r"""Rescale only when the values approach underflow.

    Rescaling every position is the textbook fix and costs a division
    per cell; firing only near the floor is the 20-fold saving.
    """
    v = [float(q) for q in k.vec(values)]
    if not v:
        raise ValueError("phmmsr: nothing to rescale")
    m = max(abs(q) for q in v)
    if m > float(floor):
        return {"values": v, "rescaled": False, "factor": 1.0,
                "log_offset": 0.0}
    if m <= 0.0:
        raise ValueError("phmmsr: the whole vector underflowed; the "
                         "rescale interval is too long")
    f = float(target) / m
    return {"values": [q * f for q in v], "rescaled": True,
            "factor": f, "log_offset": math.log(f),
            "note": "the log offset must be accumulated or the final "
                    "score is wrong by exactly this much"}


def search_pipeline(sequences, profile, msv_threshold=0.02,
                    mu=10.0, lam=0.7, full_score=None):
    r"""MSV filter first, full Forward/Backward on survivors.

    A filter that discards true positives is not an acceleration; the
    survivor fraction and the discarded scores are both reported.
    """
    passed, scores, discarded = [], [], []
    for idx, s in enumerate(sequences):
        m = msv_score(s, profile)["score"]
        p = gumbel_pvalue(m, mu, lam)
        scores.append((idx, m, p))
        if p <= float(msv_threshold):
            passed.append(idx)
        else:
            discarded.append((idx, m, p))
    full = {}
    if full_score is not None:
        for i in passed:
            full[i] = float(full_score(sequences[i], profile))
    return RichResult(payload={
        "estimate": passed, "passed": passed,
        "msv_scores": scores, "discarded": discarded,
        "survivor_fraction": len(passed) / float(len(sequences))
        if sequences else 0.0,
        "full_scores": full,
        "method": "HMMER3 acceleration pipeline; Eddy (2011)",
        "note": "the threshold is a P-VALUE, because MSV scores share "
                "the gapped-alignment distribution",
    })


def cheatsheet():
    return ("phmmsr: profile HMMs are more sensitive and were far "
            "slower, so accelerate with a PIPELINE. The MSV filter "
            "sums multiple UNGAPPED local segments in a striped "
            "vector layout -- dropping gaps is what makes it "
            "vectorisable -- and its scores follow the SAME Gumbel "
            "distribution as gapped local alignment scores, so the "
            "filter threshold is a P-VALUE rather than an arbitrary "
            "cutoff. SPARSE RESCALING fires only near underflow "
            "instead of at every cell, for 20x on Forward/Backward. "
            "Survivors get the full model; a filter that loses true "
            "positives is a worse method, not a faster one.")


# compact alias per ledger/NAMING.md
hmmersearch = search_pipeline
