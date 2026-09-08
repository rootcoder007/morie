# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Profile hidden Markov model: forward likelihood and Viterbi alignment.

Eddy (1998), "Profile hidden Markov models", Bioinformatics
14(9):755-763, doi:10.1093/bioinformatics/14.9.755, and Krogh,
Brown, Mian, Sjolander and Haussler (1994), J. Mol. Biol.
235(5):1501-1531, doi:10.1006/jmbi.1994.1104, for the architecture:
each profile position j has a match state M_j with its own emission
distribution, an insert state I_j emitting from a background
distribution, and a silent delete state D_j.  The transitions used
here are the standard seven, M->M, M->I, M->D, I->M, I->I, D->M and
D->D.  The forward recursion sums over alignments and Viterbi takes
the maximum; the delete states are silent, so their recursion consumes
no residue.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["hmm_profile"]

_NEG = float("-inf")


def _lg(p):
    return math.log(p) if p > 0 else _NEG


def _lse(a, b):
    if a == _NEG:
        return b
    if b == _NEG:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def hmm_profile(seq, profile):
    """Forward log-likelihood and Viterbi log-odds of a sequence against a profile.

    Parameters
    ----------
    seq : array-like of int
        Residue indices, 0-based.
    profile : object with entries
        match  : L x A emission probabilities,
        insert : length-A background emission probabilities,
        trans  : dict-like with mm, mi, md, im, ii, dm, dd.
    """
    xs = [int(v) for v in core.vec(seq)]
    T = len(xs)
    if T == 0:
        raise ValueError("hmm_profile: seq is empty")
    get = (lambda k: profile.get(k)) if hasattr(profile, "get") else (lambda k: getattr(profile, k, None))
    Mm = core.mat(get("match"))
    ins = core.vec(get("insert"))
    tr = get("trans")
    tget = (lambda k: tr.get(k)) if hasattr(tr, "get") else (lambda k: getattr(tr, k))
    L = len(Mm)
    A = len(Mm[0])
    if L == 0:
        raise ValueError("hmm_profile: profile has no positions")
    if len(ins) != A:
        raise ValueError("hmm_profile: insert distribution has the wrong alphabet size")
    for v in xs:
        if v < 0 or v >= A:
            raise ValueError("hmm_profile: residue index out of range")
    lmm, lmi, lmd = _lg(float(tget("mm"))), _lg(float(tget("mi"))), _lg(float(tget("md")))
    lim, lii = _lg(float(tget("im"))), _lg(float(tget("ii")))
    ldm, ldd = _lg(float(tget("dm"))), _lg(float(tget("dd")))
    NEG = _NEG
    fM = [[NEG] * (L + 1) for _ in range(T + 1)]
    fI = [[NEG] * (L + 1) for _ in range(T + 1)]
    fD = [[NEG] * (L + 1) for _ in range(T + 1)]
    vM = [[NEG] * (L + 1) for _ in range(T + 1)]
    vI = [[NEG] * (L + 1) for _ in range(T + 1)]
    vD = [[NEG] * (L + 1) for _ in range(T + 1)]
    fM[0][0] = 0.0
    vM[0][0] = 0.0
    for i in range(T + 1):
        for j in range(L + 1):
            if i == 0 and j == 0:
                continue
            if i > 0 and j > 0:
                e = _lg(Mm[j - 1][xs[i - 1]])
                cand = [fM[i - 1][j - 1] + lmm, fI[i - 1][j - 1] + lim, fD[i - 1][j - 1] + ldm]
                acc = NEG
                for cv in cand:
                    acc = _lse(acc, cv)
                fM[i][j] = acc + e
                best = max([vM[i - 1][j - 1] + lmm, vI[i - 1][j - 1] + lim, vD[i - 1][j - 1] + ldm])
                vM[i][j] = best + e
            if i > 0:
                e = _lg(ins[xs[i - 1]])
                acc = _lse(fM[i - 1][j] + lmi, fI[i - 1][j] + lii)
                fI[i][j] = acc + e
                vI[i][j] = max(vM[i - 1][j] + lmi, vI[i - 1][j] + lii) + e
            if j > 0:
                acc = _lse(fM[i][j - 1] + lmd, fD[i][j - 1] + ldd)
                fD[i][j] = acc
                vD[i][j] = max(vM[i][j - 1] + lmd, vD[i][j - 1] + ldd)
    fwd = NEG
    for v in (fM[T][L], fI[T][L], fD[T][L]):
        fwd = _lse(fwd, v)
    vit = max(vM[T][L], vI[T][L], vD[T][L])
    bg = 0.0
    for v in xs:
        bg += _lg(ins[v])
    return RichResult(
        title="Profile HMM",
        summary_lines=[("length", T), ("profile positions", L)],
        payload={
            "estimate": fwd,
            "forward_logprob": fwd,
            "viterbi_logprob": vit,
            "log_odds": fwd - bg,
            "background_logprob": bg,
            "n": T,
            "method": "forward and Viterbi over match/insert/delete states, Eddy (1998); Krogh et al. (1994)",
        },
    )


def cheatsheet():
    return "hmmprf: profile hidden Markov model"


# compact alias per ledger/NAMING.md
hmmprofile = hmm_profile
