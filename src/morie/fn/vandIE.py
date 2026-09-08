# morie.fn -- function file (rootcoder007/morie)
"""Four-way decomposition reported as proportions."""

from . import _s04core as S

from ._richresult import RichResult

__all__ = ["vanderweele_decomposition"]


def vanderweele_decomposition(Y, X, M, Cc=None, a=1.0, astar=0.0, m=0.0):
    """The four-way decomposition expressed as shares of the total.

    The absolute components answer how much; the proportions answer how
    much of what happened.  The paper is explicit that the proportions
    only make sense when the components all point the same way -- with
    mixed signs a share can exceed one or go negative -- so the sign
    agreement is checked and reported rather than left for the reader to
    discover.

    Formula: ``TE = CDE + INTref + INTmed + PIE``, reported as
    ``E[CDE]/E[TE]``, ``E[INTref]/E[TE]``, ``E[INTmed]/E[TE]`` and
    ``E[PIE]/E[TE]``.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome.
    X : array-like, shape (n,)
        Exposure.
    M : array-like, shape (n,)
        Mediator.
    Cc : array-like, optional
        Covariates; read at their means.
    a, astar, m : float
        Exposure contrast and controlled mediator level.

    Returns
    -------
    RichResult
        ``estimate`` (proportion eliminated), ``p_cde``, ``p_intref``,
        ``p_intmed``, ``p_pie``, ``p_mediated``, ``p_interaction``,
        ``te``, ``same_sign``, ``n``.

    References
    ----------
    VanderWeele, T. J. (2014).  A unification of mediation and
    interaction: a 4-way decomposition.  Epidemiology 25:749-761.
    Fetched and read; the regression expressions used here are printed
    in that paper as
    ``E[CDE|c] = th1 (a - a*)``,
    ``E[INTref|c] = th3 (b0 + b1 a* + b2'c)(a - a*)``,
    ``E[INTmed|c] = th3 b1 (a - a*)(a - a*)`` and
    ``E[PIE|c] = (th2 b1 + th3 b1 a*)(a - a*)``.
    """
    theta, beta, cbar = S.medmodels(Y, X, M, Cc)
    cde, intref, intmed, pie, te = S.fourway(theta, beta, cbar, a, astar, m)
    parts = [cde, intref, intmed, pie]
    same = 1.0 if all((v >= 0.0) == (parts[0] >= 0.0) for v in parts) else 0.0
    def sh(v):
        return v / te if te != 0.0 else float("nan")
    return RichResult(payload={
        "estimate": sh(intref + intmed + pie), "p_cde": sh(cde),
        "p_intref": sh(intref), "p_intmed": sh(intmed), "p_pie": sh(pie),
        "p_mediated": sh(intmed + pie), "p_interaction": sh(intref + intmed),
        "te": te, "same_sign": same, "n": len(list(Y)),
        "method": "VanderWeele four-way decomposition, proportions"})


def cheatsheet():
    return "vandIE: Four-way decomposition reported as proportions."
