# SPDX-License-Identifier: AGPL-3.0-or-later
"""FIMO-style PWM motif scan with exact Staden p-values."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["motfom", "motif_fimo"]

_ALPHABET = "ACGT"


def _llr_matrix(pwm, background, pseudo):
    w = len(pwm)
    llr = []
    for j in range(w):
        row = []
        tot = sum(float(pwm[j][a]) for a in range(4)) + 4.0 * pseudo
        for a in range(4):
            pa = (float(pwm[j][a]) + pseudo) / tot
            if pa <= 0.0 or background[a] <= 0.0:
                raise ValueError("zero probability; use a pseudocount")
            row.append(math.log(pa / background[a], 2.0))
        llr.append(row)
    return llr


def _staden_pdist(llr, background, scale):
    # Staden (1994) dynamic programming: discretize the per-position
    # log-odds scores to an integer grid (unit 1/scale bits), then
    # convolve the four-letter score distributions across positions
    # under the zero-order background model. Returns (offset, pdf)
    # with pdf[s - offset] = P(total integer score = s).
    w = len(llr)
    illr = [[int(round(llr[j][a] * scale)) for a in range(4)]
            for j in range(w)]
    lo = sum(min(illr[j]) for j in range(w))
    hi = sum(max(illr[j]) for j in range(w))
    pdf = {0: 1.0}
    for j in range(w):
        nxt = {}
        for s, pr in pdf.items():
            for a in range(4):
                key = s + illr[j][a]
                nxt[key] = nxt.get(key, 0.0) + pr * background[a]
        pdf = nxt
    dense = [0.0] * (hi - lo + 1)
    for s, pr in pdf.items():
        dense[s - lo] = pr
    return lo, dense, illr


def motfom(sequence, pwm, background=None, pseudocount=0.0, scale=1000):
    """
    Scan a sequence with a position weight matrix, FIMO style.

    For each position i of the sequence, the log-likelihood ratio
    score of the length-w window is

        score(i) = sum_{j=1}^{w} log2( p(b_{i+j-1}, j) / bg(b) ),

    where p(b, j) is the motif probability of base b at column j and
    bg(b) the zero-order background frequency. Each score is
    converted to a p-value P(S >= score) under the null model in
    which sequences are generated independently from bg, using the
    Staden (1994) dynamic-programming distribution of the discretized
    score (integer grid of 1/scale bits), exactly as FIMO does
    (Grant, Bailey and Noble 2011, Methods paragraph 2). Scanning
    here is single-strand.

    Parameters
    ----------
    sequence : str
        DNA sequence over ACGT.
    pwm : array-like, shape (w, 4)
        Motif probability matrix, columns ordered A, C, G, T.
    background : array-like of 4 floats, optional
        Zero-order background frequencies (default uniform 0.25).
    pseudocount : float
        Added to every pwm cell before normalization.
    scale : int
        Discretization: scores are rounded to multiples of 1/scale
        bits for the p-value dynamic programming.

    Returns
    -------
    result : RichResult
        Keys: scores (per starting position, in bits), pvalues,
        best_score, best_pvalue, best_position (0-based), width,
        n_windows, method.

    References
    ----------
    Grant, C. E., Bailey, T. L. and Noble, W. S. (2011), "FIMO:
    scanning for occurrences of a given motif", Bioinformatics 27(7),
    1017-1018 (log-likelihood ratio score and dynamic-programming
    p-values, Methods). Staden, R. (1994), "Searching for motifs in
    nucleic acid sequences", Methods in Molecular Biology 25,
    93-102 (score-distribution dynamic programming). Local source:
    library/pdf/fetched-wave3/Grant-2011-FIMO-Bioinformatics.pdf.
    """
    seq = str(sequence).upper()
    P = np.asarray(pwm, dtype=float)
    if P.ndim != 2 or P.shape[1] != 4:
        raise ValueError("pwm must be (w, 4) with columns A, C, G, T")
    w = P.shape[0]
    if background is None:
        bg = [0.25, 0.25, 0.25, 0.25]
    else:
        bg = [float(b) for b in background]
        tot = sum(bg)
        bg = [b / tot for b in bg]
    pwm_rows = [[float(P[j, a]) for a in range(4)] for j in range(w)]
    llr = _llr_matrix(pwm_rows, bg, float(pseudocount))
    lo, dense, illr = _staden_pdist(llr, bg, int(scale))
    # survival function on the integer grid
    surv = [0.0] * (len(dense) + 1)
    for s in range(len(dense) - 1, -1, -1):
        surv[s] = surv[s + 1] + dense[s]
    idx = {c: a for a, c in enumerate(_ALPHABET)}
    n_win = len(seq) - w + 1
    if n_win < 1:
        raise ValueError("sequence shorter than motif")
    scores = []
    pvals = []
    for i in range(n_win):
        s_bits = 0.0
        s_int = 0
        ok = True
        for j in range(w):
            c = seq[i + j]
            if c not in idx:
                ok = False
                break
            s_bits += llr[j][idx[c]]
            s_int += illr[j][idx[c]]
        if not ok:
            scores.append(float("nan"))
            pvals.append(float("nan"))
            continue
        scores.append(s_bits)
        pvals.append(surv[s_int - lo])
    best = None
    for i in range(n_win):
        if pvals[i] == pvals[i] and (best is None or scores[i] > scores[best]):
            best = i
    if best is None:
        raise ValueError("no scorable window (non-ACGT sequence)")
    return RichResult(payload={
        "scores": np.asarray(scores),
        "pvalues": np.asarray(pvals),
        "best_score": scores[best],
        "best_pvalue": pvals[best],
        "best_position": best,
        "width": w,
        "n_windows": n_win,
        "method": "FIMO PWM scan, Staden DP p-values (Grant et al. 2011)",
    })


motif_fimo = motfom
motiffimo = motfom


def cheatsheet():
    return ("motfom(sequence, pwm) -> log2-odds PWM scores per position "
            "with exact Staden dynamic-programming p-values.")
