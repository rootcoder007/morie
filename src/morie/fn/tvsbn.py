# morie.fn -- function file (rootcoder007/morie)
"""Tversky similarity on binary fingerprints."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tversky_similarity"]


def tversky_similarity(fp_a, fp_b, alpha=0.5, beta=0.5):
    """Asymmetric set similarity with separate penalties per side.

    Tversky argument was that human similarity judgements are not
    symmetric -- a son resembles his father more than the father
    resembles the son -- and a symmetric coefficient cannot express
    that.  The two weights let the two kinds of mismatch cost different
    amounts.  Setting both to 1 recovers Jaccard-Tanimoto and both to
    0.5 recovers Dice, so the classical coefficients are the symmetric
    corners of the same family.

    Formula: ``S = |A n B| / (|A n B| + alpha |A \ B| + beta |B \ A|)``.

    Parameters
    ----------
    fp_a : array-like
        Binary fingerprint A; non-zero counts as present.
    fp_b : array-like
        Binary fingerprint B, same length.
    alpha : float, default 0.5
        Weight on features unique to A.
    beta : float, default 0.5
        Weight on features unique to B.

    Returns
    -------
    RichResult
        ``estimate``, ``common``, ``only_a``, ``only_b``, ``n_bits``.

    References
    ----------
    Tversky, A. (1977).  Features of similarity.  Psychological Review
    84:327-352, equation (5) -- the contrast model with the ratio form
    in section 4.
    """
    a = C.vec(fp_a)
    b = C.vec(fp_b)
    common = sum(1.0 for x, y in zip(a, b) if x != 0.0 and y != 0.0)
    only_a = sum(1.0 for x, y in zip(a, b) if x != 0.0 and y == 0.0)
    only_b = sum(1.0 for x, y in zip(a, b) if x == 0.0 and y != 0.0)
    den = common + alpha * only_a + beta * only_b
    s = common / den if den > 0.0 else float("nan")
    return RichResult(payload={
        "estimate": s, "common": common, "only_a": only_a, "only_b": only_b,
        "n_bits": len(a), "method": "Tversky similarity index"})


def cheatsheet():
    return "tvsbn: Tversky similarity on binary fingerprints."
