# morie.fn -- function file (rootcoder007/morie)
"""Chain (mediation) structure A->B->C: information flows, B is mediator."""

from ._richresult import RichResult
from .frkst import fork_structure

__all__ = ["chain_structure"]


def chain_structure(A, B, C, alpha=0.01):
    r"""Test the independence signature of a chain A -> B -> C.

    A chain implies exactly the same observational signature as a fork
    (they are Markov equivalent): A and C dependent marginally,
    independent given the middle node,

    .. math:: A \not\perp C, \qquad A \perp C \mid B.

    Delegates to the fork test; only the interpretation differs (B is
    a mediator here, a confounder there), and observational data alone
    cannot tell the two apart.

    Parameters
    ----------
    A, B, C : array-like, shape (n,)
        Observations along the putative chain.
    alpha : float, default 0.01
        Test level.

    Returns
    -------
    RichResult
        Same statistics as :func:`morie.fn.frkst.fork_structure`, with
        ``consistent_with_chain`` in place of ``consistent_with_fork``.

    References
    ----------
    Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University
    Press. Sec. 1.2.3 (chains block on the mediator) and Thm 1.2.8
    (observational equivalence).
    """
    out = fork_structure(A, B, C, alpha=alpha)
    payload = dict(out)
    payload["consistent_with_chain"] = payload.pop("consistent_with_fork")
    payload["method"] = "Chain independence signature: A ~ C marginally, A _||_ C | B"
    return RichResult(payload=payload)


def cheatsheet():
    return "chstr: chain A->B->C -- same Markov signature as a fork; blocked on B"
