# morie.fn -- function file (rootcoder007/morie)
"""Parallel multiple mediators (alias of :mod:`mcausm`)."""

from .mcausm import multi_mediator_causal

from ._richresult import RichResult

__all__ = ["multiple_mediators", "multiplemediators"]


def multiple_mediators(Y, X, M_list, C=None):
    """Specific and joint natural indirect effects through parallel mediators.

    This module is an ALIAS.  The regressions are implemented once, in
    ``mcausm.multi_mediator_causal``; this entry point supplies the
    argument order used in the mediation literature (outcome first) and
    delegates.  No second copy of the arithmetic exists.

    Each mediator is regressed on the exposure,
    ``M_k = alpha_0k + a_k X + ...``, and the outcome on the exposure
    and ALL mediators at once,
    ``Y = beta_0 + c' X + sum_k b_k M_k + ...``.  The specific indirect
    effect through mediator ``k`` is ``a_k b_k`` and the joint indirect
    effect is ``sum_k a_k b_k``.

    Fitting the mediators one at a time and the outcome on all of them
    is what makes these "parallel": no mediator is allowed to cause
    another.  Daniel et al. (2015) is the reference for what that
    assumption buys and costs -- with mediator-mediator causation the
    ``a_k b_k`` products are no longer the natural indirect effects, and
    the sum no longer decomposes the total effect.  This routine
    computes the parallel-model quantities; it does not test the
    assumption.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome.
    X : array-like, shape (n,)
        Exposure.
    M_list : array-like, shape (n, k)
        Mediators, one column each.
    C : array-like, optional
        Baseline covariates.

    Returns
    -------
    RichResult
        ``indirect`` (per mediator), ``indirect_total``, ``direct``,
        ``total``, ``a``, ``b``, ``k``, ``n``.

    References
    ----------
    Daniel, R. M., De Stavola, B. L., Cousens, S. N. and Vansteelandt,
    S. (2015), "Causal mediation analysis with multiple mediators",
    Biometrics 71(1), 1-14, doi:10.1111/biom.12248, verified against
    Crossref.  VanderWeele, T. J. (2015), Explanation in Causal
    Inference, Oxford University Press, ch. 5, for the specific-versus-
    joint indirect effect distinction.  Neither source was in the local
    corpus; the parallel-mediator products above are the standard
    published form and are stated in full.
    """
    r = multi_mediator_causal(X, M_list, Y, C)
    return RichResult(payload={
        "indirect": list(r["indirect"]), "indirect_total": r["indirect_total"],
        "direct": r["direct"], "total": r["total"],
        "a": list(r["a"]), "b": list(r["b"]), "k": r["k"], "n": r["n"],
        "method": "Parallel multiple mediators (specific and joint NIE)"})


multiplemediators = multiple_mediators


def cheatsheet():
    return "multM: parallel multiple mediators (alias of mcausm)"
