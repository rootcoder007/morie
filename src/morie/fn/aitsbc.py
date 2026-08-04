# morie.fn -- function file (rootcoder007/morie)
"""Subcomposition formed from a selected set of parts."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['compsubcomp', 'aitchison_subcomposition']


def compsubcomp(x, parts, total=1.0):
    """Subcomposition formed from a selected set of parts.

    Formula: sub(x; S) = C( x_i : i in S )

    Parameters
    ----------
    x : array-like
        Composition with strictly positive parts.
    parts : array-like of int
        1-based indices of the parts to keep; at least two, no repeats.
    total : float
        Constant kappa the closure sums to.

    Returns
    -------
    RichResult
        ``composition``, ``parts``, ``total``, ``D``, ``D_full``.

    References
    ----------
    Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read; see EXTERNAL_SOURCES.md.  A subcomposition is the closure of a selected subset of parts.  Log-ratios between retained parts are unchanged by the operation, which is the property that makes subcompositional coherence a requirement on compositional methods.  Indices are 1-based in BOTH language arms.  Implemented in the standard published form.  The log-ratio algebra it rests on was verified against Mateu-Figueras, Pawlowsky-Glahn and Egozcue, arXiv:0802.2643 Sect. 4.1 (fetched and archived), but this particular definition is not printed there and could not be checked against Aitchison's own text.
    """
    x = C.vec(x)
    D0 = len(x)
    if any(v <= 0.0 for v in x):
        raise ValueError("compositions must be strictly positive")
    idx = [int(v) for v in parts]
    if len(idx) < 2:
        raise ValueError("a subcomposition needs at least two parts")
    if len(set(idx)) != len(idx):
        raise ValueError("parts must not repeat")
    if any(not 1 <= i <= D0 for i in idx):
        raise ValueError("parts must be 1-based indices into x")
    sub = [x[i - 1] for i in idx]
    s = sum(sub)
    k = float(total)
    return RichResult(payload={
        "composition": [k * v / s for v in sub], "parts": idx, "total": k,
        "D": len(idx), "D_full": D0, "method": "Subcomposition"})


aitchison_subcomposition = compsubcomp


def cheatsheet():
    return 'aitsbc: Subcomposition formed from a selected set of parts.'
