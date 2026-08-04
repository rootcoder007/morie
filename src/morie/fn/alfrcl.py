# morie.fn -- function file (rootcoder007/morie)
"""Recycling training objective of AlphaFold."""

from __future__ import annotations

from ._richresult import RichResult

__all__ = ["alphafold_recycle_loss"]


def alphafold_recycle_loss(losses, nprime=None):
    """Recycling loss -- Algorithm 31 and equation (48), pp. 42-43.

    The objective is the average loss over all recycling iterations,
    equation (48).  Training does not evaluate that average; it samples one
    iteration ``N'`` uniformly and trains only that one, stopping gradients
    from flowing into the earlier iterations, and skipping the later ones
    entirely.

    That sampled estimate is unbiased, which is the whole justification for
    the scheme: averaging the single-iteration estimate over every possible
    ``N'`` returns equation (48) exactly.  Nothing is sampled here -- pass
    ``nprime`` to get one iteration's estimate, or omit it for the full
    average.

    Parameters
    ----------
    losses : list of float
        Loss at each recycling iteration, ``losses[c]`` for iteration
        ``c + 1``.
    nprime : int, optional
        One-based iteration selected by the training procedure.  When
        given, the estimate is ``losses[nprime - 1]``.

    Returns
    -------
    result : RichResult
        Keys: ``estimate`` (the selected iteration's loss, or the average),
        ``average`` (equation 48), ``ncycle``, ``nprime``, ``expected``
        (the mean of the estimator over all ``N'``, which equals
        ``average``), ``method``.

    Notes
    -----
    The parity harness checks the unbiasedness identity directly: the mean
    over ``N' = 1 ... Ncycle`` of the single-iteration estimate equals the
    equation (48) average exactly.  That is a published claim about the
    estimator, not a restatement of this code.

    References
    ----------
    Jumper et al (2021) Nature 596:583-589, Supplementary Algorithm 31,
    equation (48)
    """
    nc = len(losses)
    if nc == 0:
        raise ValueError("losses must not be empty")
    avg = sum(losses) / nc
    if nprime is not None and not 1 <= nprime <= nc:
        raise ValueError("nprime %r outside 1..%d" % (nprime, nc))
    est = avg if nprime is None else losses[nprime - 1]
    return RichResult(
        payload={
            "estimate": est,
            "average": avg,
            "expected": sum(losses) / nc,
            "ncycle": nc,
            "nprime": nprime,
            "method": "AlphaFold recycling training objective",
        }
    )


def cheatsheet():
    return "alfrcl: recycling objective, averaged and single-iteration forms"
