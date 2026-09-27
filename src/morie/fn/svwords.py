"""Wordscores: scaling texts from reference texts with known positions (Laver, Benoit and Garry 2003).

Laver, M., Benoit, K. and Garry, J. (2003). Extracting policy positions from political texts
using words as data. American Political Science Review 97, 311-331.
"""

import math

from ._richresult import RichResult

__all__ = ["wordscores"]


def wordscores(ref_counts, ref_scores, virgin_counts, rescale=True):
    r"""Score virgin texts from word counts.

    With F_wr the relative frequency of word w in reference text r, P_wr = F_wr / sum_r F_wr
    and the word score S_w = sum_r P_wr A_r for reference positions A_r. A virgin text v scores
    S_v = sum_w F_wv S_w over its scored words (those seen in the references), with
    V_v = sum_w F_wv (S_w - S_v)^2 and standard error sqrt(V_v / N_v), N_v its number of scored
    words. The LBG rescaling S*_v = (S_v - mean S_v) SD_A / SD_Sv + mean S_v restores the
    spread of the reference scores (Laver, Benoit and Garry 2003, eqs. 1-6).

    Parameters
    ----------
    ref_counts : reference texts x words counts
    ref_scores : sequence of reference positions A_r
    virgin_counts : virgin texts x words counts (same word columns)
    rescale : bool

    Returns
    -------
    RichResult
        Keys: word_scores (None for unseen words), raw, se, rescaled (with ``rescale`` and at
        least two virgin texts).

    References
    ----------
    Laver, M., Benoit, K. and Garry, J. (2003). American Political Science Review 97, 311-331.

    Examples
    --------
    >>> r = wordscores([[2, 0, 2], [0, 2, 2]], [-1, 1], [[1, 1, 2]], rescale=False)
    >>> r["raw"]
    [0.0]
    """
    R = [[float(v) for v in r] for r in ref_counts]
    A = [float(v) for v in ref_scores]
    Vc = [[float(v) for v in r] for r in virgin_counts]
    W = len(R[0])
    F = [[c / sum(r) for c in r] for r in R]
    S = []
    for w in range(W):
        col = [F[r][w] for r in range(len(R))]
        tot = sum(col)
        S.append(None if tot == 0 else sum(col[r] / tot * A[r] for r in range(len(R))))
    raw, se = [], []
    for v in Vc:
        seen = [w for w in range(W) if S[w] is not None and v[w] > 0]
        N = sum(v[w] for w in seen)
        fv = {w: v[w] / N for w in seen}
        sv = sum(fv[w] * S[w] for w in seen)
        var = sum(fv[w] * (S[w] - sv) ** 2 for w in seen)
        raw.append(sv)
        se.append(math.sqrt(var / N))
    out = {"word_scores": S, "raw": raw, "se": se}
    if rescale and len(raw) > 1:
        mv = sum(raw) / len(raw)
        sdv = math.sqrt(sum((x - mv) ** 2 for x in raw) / (len(raw) - 1))
        ma = sum(A) / len(A)
        sda = math.sqrt(sum((a - ma) ** 2 for a in A) / (len(A) - 1))
        out["rescaled"] = [(x - mv) * sda / sdv + mv for x in raw]
    return RichResult(title="Wordscores", summary_lines=[("virgin texts", len(raw))], payload=out)


def cheatsheet():
    return "svwords: Wordscores text scaling with LBG rescaling"
