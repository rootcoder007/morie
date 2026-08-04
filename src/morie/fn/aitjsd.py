# morie.fn -- function file (rootcoder007/morie)
"""Jensen-Shannon divergence between two closed compositions."""

from __future__ import annotations

from .jsdivg import jensen_shannon_divergence

__all__ = ["compositional_jsd"]


def compositional_jsd(p, q, base=2.0):
    """Jensen-Shannon divergence for compositional data.

    A composition carries only relative information, so it is closed to
    the simplex before the divergence is taken -- which is exactly what
    :func:`morie.fn.jsdivg.jensen_shannon_divergence` does with
    ``normalize=True``, and why the arithmetic is not duplicated here.
    Closure makes the answer invariant to the total, so parts per
    million and proportions give the same number.

    Note that this is an information divergence on the closed parts, not
    an Aitchison distance: it is not invariant to subcomposition and it
    does not use log-ratios.  For a zero part the JSD term is simply
    zero, where a log-ratio geometry would be undefined -- which is a
    reason to prefer it on sparse compositions and a reason not to call
    it Aitchison.

    Parameters
    ----------
    p, q : array-like
        Non-negative compositions over the same parts.
    base : float
        Logarithm base; 2 gives bits.

    Returns
    -------
    RichResult
        As :func:`jensen_shannon_divergence`.

    References
    ----------
    Lin (1991), IEEE Transactions on Information Theory 37:145-151;
    coded form from ``philentropy``.  See ``morie.fn.jsdivg`` for the
    full note.
    """
    return jensen_shannon_divergence(p, q, base=base, normalize=True)


def cheatsheet():
    return "compositional_jsd(p, q): JSD on closed compositions."


# compact alias per ledger/NAMING.md
compjsd = compositional_jsd
