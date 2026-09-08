# SPDX-License-Identifier: AGPL-3.0-or-later
"""METEOR machine-translation metric (Banerjee-Lavie 2005)."""

from ._richresult import RichResult

__all__ = ["meteor", "meteor_score"]


def _match(cand, ref):
    # Exact-match stage: each candidate unigram maps to at most one
    # reference unigram of identical surface form; candidate tokens
    # are processed left to right and take the earliest unmatched
    # occurrence in the reference (deterministic; pinned in both
    # arms). This attains the maximum number of unigram matches for
    # the exact module.
    used = [False] * len(ref)
    pairs = []
    for i, w in enumerate(cand):
        for j, r in enumerate(ref):
            if not used[j] and r == w:
                used[j] = True
                pairs.append((i, j))
                break
    return pairs


def _chunks(pairs):
    # Fewest chunks such that unigrams in each chunk are adjacent in
    # the candidate AND mapped to adjacent unigrams in the reference.
    if not pairs:
        return 0
    ch = 1
    for k in range(1, len(pairs)):
        if not (pairs[k][0] == pairs[k - 1][0] + 1
                and pairs[k][1] == pairs[k - 1][1] + 1):
            ch += 1
    return ch


def meteor(candidate, reference, lowercase=True):
    """
    METEOR sentence score against a single reference translation.

    Unigrams of the candidate are aligned to the reference by exact
    match. With m matched unigrams, unigram precision P = m / |cand|
    and recall R = m / |ref| combine as the recall-weighted harmonic
    mean

        Fmean = 10 P R / (R + 9 P).

    Matched unigrams are grouped into the fewest possible chunks of
    contiguous candidate positions mapped to contiguous reference
    positions, giving the fragmentation penalty

        Penalty = 0.5 * (#chunks / #unigrams_matched)^3,

    and the final score is  Score = Fmean * (1 - Penalty). A
    candidate with no matches scores 0.

    Parameters
    ----------
    candidate : str or sequence of str
        System translation (string is split on whitespace).
    reference : str or sequence of str
        Reference translation.
    lowercase : bool
        Case-fold before matching (default True).

    Returns
    -------
    result : RichResult
        Keys: score, fmean, penalty, precision, recall, matches,
        chunks, len_candidate, len_reference, method.

    References
    ----------
    Banerjee, S. and Lavie, A. (2005), "METEOR: An automatic metric
    for MT evaluation with improved correlation with human
    judgments", Proceedings of the ACL Workshop on Intrinsic and
    Extrinsic Evaluation Measures for Machine Translation and/or
    Summarization, Ann Arbor, 65-72. Fmean, chunk penalty and Score
    formulas, Section 2.1, p. 68 (including the two-chunk worked
    example). Local source:
    library/pdf/fetched-wave3/Banerjee-Lavie-2005-METEOR-ACL.pdf.
    """
    cand = (candidate.split() if isinstance(candidate, str)
            else [str(w) for w in candidate])
    ref = (reference.split() if isinstance(reference, str)
           else [str(w) for w in reference])
    if lowercase:
        cand = [w.lower() for w in cand]
        ref = [w.lower() for w in ref]
    if not cand or not ref:
        raise ValueError("candidate and reference must be nonempty")
    pairs = _match(cand, ref)
    m = len(pairs)
    if m == 0:
        return RichResult(payload={
            "score": 0.0, "fmean": 0.0, "penalty": 0.0,
            "precision": 0.0, "recall": 0.0, "matches": 0,
            "chunks": 0, "len_candidate": len(cand),
            "len_reference": len(ref),
            "method": "METEOR exact-match stage (Banerjee-Lavie 2005)",
        })
    prec = m / float(len(cand))
    rec = m / float(len(ref))
    fmean = 10.0 * prec * rec / (rec + 9.0 * prec)
    ch = _chunks(pairs)
    penalty = 0.5 * (ch / float(m)) ** 3
    return RichResult(payload={
        "score": fmean * (1.0 - penalty),
        "fmean": fmean,
        "penalty": penalty,
        "precision": prec,
        "recall": rec,
        "matches": m,
        "chunks": ch,
        "len_candidate": len(cand),
        "len_reference": len(ref),
        "method": "METEOR exact-match stage (Banerjee-Lavie 2005)",
    })


meteor_score = meteor


def cheatsheet():
    return ("meteor(candidate, reference) -> Fmean = 10PR/(R+9P), "
            "Penalty = 0.5 (chunks/matches)^3, Score = Fmean (1 - Penalty).")
