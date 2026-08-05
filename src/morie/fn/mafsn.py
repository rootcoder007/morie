# morie.fn -- function file (rootcoder007/morie)
"""Rosenthal's fail-safe N."""

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ma_fail_safe_n"]


def ma_fail_safe_n(z_scores, alpha=0.05):
    """How many missing null studies would sink a significant result.

    The file-drawer count answers a question about publication bias with a
    number, which is its appeal and also its flaw: it assumes the missing
    studies average exactly zero effect, which no plausible selection
    mechanism produces.  Read it as an upper bound on fragility, never as
    evidence that bias is absent.

    Formula: ``N_fs = (sum z_i)^2 / z_alpha^2 - k`` with ``z_alpha`` the
    one-tailed critical value -- Rosenthal (1979) eq. (2).

    Parameters
    ----------
    z_scores : array-like
        One-tailed z statistics of the included studies.
    alpha : float, default 0.05
        One-tailed level.

    Returns
    -------
    RichResult
        ``Nfs``, ``z_combined`` (Stouffer's Z), ``z_alpha``, ``k``.

    References
    ----------
    Rosenthal, R. (1979).  The file drawer problem and tolerance for null
    results.  Psychological Bulletin 86(3):638-641.
    doi:10.1037/0033-2909.86.3.638.
    """
    z = [float(t) for t in core.vec(z_scores)]
    k = len(z)
    if k == 0:
        raise ValueError("no studies")
    a = float(alpha)
    if a <= 0.0 or a >= 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    za = core.qnorm(1.0 - a)
    s = sum(z)
    return RichResult(payload={
        "Nfs": s * s / (za * za) - k,
        "z_combined": s / math.sqrt(k), "z_alpha": za, "k": k,
        "method": "Rosenthal's fail-safe N"})


def cheatsheet():
    return "mafsn: Rosenthal's fail-safe N for the file-drawer problem"


# compact alias per ledger/NAMING.md
mafailsafen = ma_fail_safe_n
