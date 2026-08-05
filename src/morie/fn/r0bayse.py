# morie.fn -- function file (rootcoder007/morie)
"""Basic reproduction number -- delegates to :mod:`morie.fn.r0`.

``r0`` already carries both routes to R0 (the direct ratio and the
Newton inversion of the final-size relation).  This module is the name
the backlog asked for, not a second implementation; the R arm added
alongside it, ``R0``, is the R mirror that ``r0`` was missing, and
``R0bayse`` aliases that.
"""

from ._richresult import RichResult
from .r0 import basic_reproduction_number as _r0

__all__ = ["basic_reproduction"]


def basic_reproduction(beta=None, gamma=None, attack_rate=None,
                       tol=1e-8, max_iter=100):
    """R0 either as ``beta / gamma`` or inverted from the attack rate.

    The second route matters more than it looks: an outbreak reports a
    final attack rate, not a transmission rate, and the final-size
    relation ``1 - AR = exp(-R0 AR)`` inverts it without any dynamic
    model at all.  It is solved by Newton, which is deterministic, so
    both language arms land on the same digits.

    Formula: ``R0 = beta / gamma``; or solve ``1 - AR = exp(-R0 AR)``.

    Parameters
    ----------
    beta, gamma : float, optional
        Transmission and recovery rates; ``gamma`` must be positive.
    attack_rate : float, optional
        Final attack rate in (0, 1); used when the rates are unknown.
    tol : float, default 1e-8
        Newton convergence tolerance.
    max_iter : int, default 100
        Maximum Newton iterations.

    Returns
    -------
    RichResult
        ``estimate`` (R0), ``R0``, ``route`` (1.0 direct, 2.0 Newton).

    References
    ----------
    Diekmann, O., Heesterbeek, J. A. P. & Metz, J. A. J. (1990).  On the
    definition and the computation of the basic reproduction ratio R0 in
    models for infectious diseases in heterogeneous populations.  Journal
    of Mathematical Biology 28(4):365-382.  doi:10.1007/BF00178324.
    """
    r = _r0(beta=beta, gamma=gamma, attack_rate=attack_rate,
            tol=tol, max_iter=max_iter)
    direct = 1.0 if r.extra.get("method") == "direct" else 2.0
    return RichResult(payload={
        "estimate": float(r.estimate), "R0": float(r.estimate),
        "route": direct,
        "method": "R0 by " + str(r.extra.get("method"))})


def cheatsheet():
    return "r0bayse: Basic reproduction number R0 (delegates to r0)"
