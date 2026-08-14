"""BLAST maximal segment pair + Karlin-Altschul statistics (Altschul et al. 1990)."""

import math

from ._richresult import RichResult

__all__ = ["blastp", "blast_msp"]


def _best_local(a, b, score):
    # Smith-Waterman with NO gaps: the maximal segment pair (MSP) is
    # the ungapped local alignment of highest score (Altschul et al.
    # 1990, Sec. 2a), found by the Kadane-style ungapped scan over
    # every diagonal.
    n, m = len(a), len(b)
    best = 0
    bi = bj = ln = 0
    for d in range(-(m - 1), n):
        run = 0
        start = 0
        i = max(0, d)
        j = i - d
        pos = 0
        while i < n and j < m:
            run += score(a[i], b[j])
            if run <= 0:
                run = 0
                start = pos + 1
            elif run > best:
                best = run
                ln = pos - start + 1
                bi = max(0, d) + start
                bj = bi - d
            i += 1
            j += 1
            pos += 1
    return best, bi, bj, ln


def blastp(query, subject, match=1.0, mismatch=-1.0, K=0.1, lam=1.0,
           score_matrix=None):
    """
    Maximal segment pair score with Karlin-Altschul E-value.

    Altschul, Gish, Miller, Myers & Lipman (1990): the local
    similarity measure BLAST approximates is the MSP score -- the
    highest-scoring ungapped segment pair between two sequences (his
    Sec. 2a).  The statistical significance uses the Karlin-Altschul
    result (his Sec. 2, Eq. for E): for a comparison of sequences of
    lengths m and n the expected number of distinct segment pairs
    with score >= S is

        E = K m n exp(-lambda S),

    an extreme-value law; the probability of at least one such pair
    is 1 - exp(-E).  This routine computes the exact MSP by ungapped
    dynamic programming and reports the E-value and p-value.

    Sources
    -------
    Altschul, S. F., Gish, W., Miller, W., Myers, E. W. & Lipman,
    D. J. (1990). Basic local alignment search tool. *Journal of
    Molecular Biology*, 215(3), 403-410, Secs. 2-3 (MSP and
    Karlin-Altschul statistics) (local copy fetched-wave3/Basic
    Local Alignment Search Tool.pdf).

    Parameters
    ----------
    query, subject : str
        Sequences.
    match, mismatch : float
        Substitution scores if no score_matrix is given.
    K, lam : float
        Karlin-Altschul parameters K and lambda.
    score_matrix : dict, optional
        {(a, b): score} overriding match/mismatch.

    Returns
    -------
    RichResult
        Keys: score (MSP), e_value, p_value, q_start, s_start,
        length.
    """
    q = str(query)
    s = str(subject)
    if not q or not s:
        raise ValueError("sequences must be non-empty")
    if score_matrix is not None:
        def score(x, y):
            return float(score_matrix.get((x, y),
                         score_matrix.get((y, x), mismatch)))
    else:
        def score(x, y):
            return match if x == y else mismatch
    msp, qi, sj, ln = _best_local(q, s, score)
    m, n = len(q), len(s)
    E = float(K) * m * n * math.exp(-float(lam) * msp)
    p = 1.0 - math.exp(-E)
    return RichResult(payload={
        "score": msp,
        "e_value": E,
        "p_value": p,
        "q_start": qi,
        "s_start": sj,
        "length": ln,
        "method": "BLAST MSP + Karlin-Altschul E-value (Altschul 1990)",
    })


# long descriptive alias (stub-era name)
blast_msp = blastp


def cheatsheet():
    return "blastp: MSP = best ungapped local score; E = K m n exp(-lam S)"

# public names resolved by fn/_lazy_map.json
blast_protein = blastp
blastprotein = blastp
