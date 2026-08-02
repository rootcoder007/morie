# morie.fn -- function file (rootcoder007/morie)
"""Roll-call object: encode a vote matrix for spatial scaling."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["pscl_rollcall"]


def pscl_rollcall(vote_matrix, lop=0.025, yea=(1,), nay=(0,), missing=None):
    r"""Encode and screen a roll-call matrix the way scaling code expects.

    Recodes arbitrary yea/nay/missing codes to 1/0/NaN, then flags
    lopsided roll calls -- those whose minority share is at or below
    ``lop`` (wnominate's default 0.025) -- because a vote with almost
    no minority carries almost no spatial information and distorts
    fit statistics. Reports margins, minority sides, and the screened
    matrix.

    Parameters
    ----------
    vote_matrix : array-like, shape (n, q)
        Raw codes.
    lop : float, default 0.025
        Lopsidedness threshold on the minority share.
    yea, nay : tuples of codes counted as yea / nay.
    missing : tuple of codes treated as missing (besides NaN).

    Returns
    -------
    RichResult
        keys: ``votes`` (n, q recoded), ``keep`` (q, boolean, False =
        lopsided or empty), ``margins`` (q, yea share), ``minority``
        (q, minority side: 0/1/NaN), ``n_dropped``, ``n``, ``q``,
        ``method``.

    References
    ----------
    Poole, K. T. & Rosenthal, H. (1997). *Congress*. Oxford
    University Press. (screening lopsided votes)

    Armstrong, D. A. et al. (2014). *Analyzing Spatial Models of
    Choice and Judgment*. CRC Press. Sec. 5.3 (rollcall objects and
    the lop argument), pp. 139-144.
    """
    V_raw = np.asarray(vote_matrix)
    if V_raw.ndim != 2:
        raise ValueError("vote_matrix must be 2-D.")
    lop = float(lop)
    if not 0 <= lop < 0.5:
        raise ValueError(f"lop must lie in [0, 0.5), got {lop}.")

    V = np.full(V_raw.shape, np.nan)
    for code in yea:
        V[V_raw == code] = 1.0
    for code in nay:
        V[V_raw == code] = 0.0
    if missing is not None:
        for code in missing:
            V[V_raw == code] = np.nan

    n, q = V.shape
    margins = np.full(q, np.nan)
    minority = np.full(q, np.nan)
    keep = np.zeros(q, dtype=bool)
    for j in range(q):
        col = V[:, j]
        col = col[~np.isnan(col)]
        if col.size == 0:
            continue
        share = float(col.mean())
        margins[j] = share
        min_share = min(share, 1 - share)
        minority[j] = 1.0 if share < 0.5 else 0.0 if share > 0.5 else np.nan
        keep[j] = min_share > lop

    return RichResult(
        payload={
            "votes": V,
            "keep": keep,
            "margins": margins,
            "minority": minority,
            "n_dropped": int(q - keep.sum()),
            "n": int(n),
            "q": int(q),
            "method": f"Roll-call encoding + lopsidedness screen (lop = {lop})",
        }
    )


def cheatsheet():
    return "pscrc: recode to 1/0/NaN, drop roll calls with minority share <= lop"
