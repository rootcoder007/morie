# morie.fn -- function file (rootcoder007/morie)
"""Effective reproduction number from susceptible depletion."""

from ._richresult import RichResult

__all__ = ["effective_reproduction"]


def effective_reproduction(R0, S, N):
    """``Rt = R0 S / N``, with the growth decision that goes with it.

    This is the susceptible-depletion Rt, not the renewal-equation Rt.
    The two are routinely confused and the stub this replaces cited Cori
    et al. (2013) for the formula ``R0 S / N``, which is not the
    estimator that paper defines -- Cori's Rt comes from the incidence
    curve and a serial-interval distribution and is implemented
    elsewhere in this package as ``morie.fn.epirf.effective_rt``
    (Wallinga-Teunis) and ``Rtrenew``.  DUPMAP.tsv lists this module as
    a duplicate of ``epirf``; it is not, for the same reason.

    The herd-immunity threshold falls straight out: ``Rt`` crosses one
    when the susceptible fraction falls to ``1 / R0``, so
    ``1 - 1 / R0`` is reported alongside.

    Formula: ``Rt = R0 S / N``; growing iff ``Rt > 1``.

    Parameters
    ----------
    R0 : float
        Basic reproduction number, non-negative.
    S : float
        Current susceptible count, in [0, N].
    N : float
        Population size, positive.

    Returns
    -------
    RichResult
        ``estimate`` (Rt), ``Rt``, ``growing`` (1.0 when ``Rt > 1``),
        ``susceptible_fraction``, ``herd_immunity_threshold``.

    References
    ----------
    Diekmann, O., Heesterbeek, J. A. P. & Metz, J. A. J. (1990).  On the
    definition and the computation of the basic reproduction ratio R0 in
    models for infectious diseases in heterogeneous populations.  Journal
    of Mathematical Biology 28(4):365-382.  doi:10.1007/BF00178324.
    """
    R0 = float(R0)
    S = float(S)
    N = float(N)
    if R0 < 0.0:
        raise ValueError("effective_reproduction: R0 must be non-negative")
    if N <= 0.0:
        raise ValueError("effective_reproduction: N must be positive")
    if S < 0.0 or S > N:
        raise ValueError("effective_reproduction: S must lie in [0, N]")
    frac = S / N
    rt = R0 * frac
    return RichResult(payload={
        "estimate": rt, "Rt": rt, "growing": 1.0 if rt > 1.0 else 0.0,
        "susceptible_fraction": frac,
        "herd_immunity_threshold": 1.0 - 1.0 / R0 if R0 > 0.0 else float("nan"),
        "method": "Effective reproduction number Rt = R0 S / N"})


def cheatsheet():
    return "reffec: Effective reproduction number Rt"
