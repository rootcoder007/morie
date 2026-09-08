# morie.fn -- function file (rootcoder007/morie)
"""Orwin's fail-safe N."""

from ._richresult import RichResult

__all__ = ["ma_orwin_fsn"]


def ma_orwin_fsn(d_obs, d_crit, d_filldraw, k):
    """How many missing studies would drag the effect down to trivial.

    Rosenthal's count asks how many nulls would make the result
    non-significant, which ties the answer to the sample size rather than
    to the effect.  Orwin's asks how many would push the pooled effect
    below whatever magnitude the reader considers trivial, and it lets the
    missing studies carry a non-zero effect of their own -- both of which
    make the number mean something substantive rather than procedural.

    Formula: ``N_fs = k (d_obs - d_crit)/(d_crit - d_fill)`` -- Orwin
    (1983) eq. (2).

    Parameters
    ----------
    d_obs : float
        Observed pooled effect.
    d_crit : float
        Effect size considered trivial.
    d_filldraw : float
        Mean effect assumed for the unretrieved studies.
    k : int
        Number of studies in the meta-analysis.

    Returns
    -------
    RichResult
        ``Nfs``, ``Nfs_ceiling`` (rounded up to a whole study),
        ``d_obs``, ``d_crit``, ``d_fill``, ``k``.

    References
    ----------
    Orwin, R. G. (1983).  A fail-safe N for effect size in meta-analysis.
    Journal of Educational Statistics 8(2):157-159.  doi:10.2307/1164923.
    """
    do = float(d_obs)
    dc = float(d_crit)
    df = float(d_filldraw)
    kk = float(k)
    if kk < 1.0:
        raise ValueError("k must be at least one")
    if abs(dc - df) < 1e-15:
        raise ValueError("d_crit and d_filldraw must differ")
    n = kk * (do - dc) / (dc - df)
    ceil = float(int(n)) + (1.0 if n > float(int(n)) else 0.0)
    return RichResult(payload={
        "Nfs": n, "Nfs_ceiling": ceil, "d_obs": do, "d_crit": dc,
        "d_fill": df, "k": kk,
        "method": "Orwin's fail-safe N"})


def cheatsheet():
    return "maorw: Orwin's fail-safe N against a trivial-effect target"


# compact alias per ledger/NAMING.md
maorwinfsn = ma_orwin_fsn
