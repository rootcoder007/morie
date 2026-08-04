# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Monte Carlo dropout predictive average.

Geron, A. (2026). Hands-On Machine Learning with Scikit-Learn and PyTorch. O'Reilly, ch. 11 p. 410
"""

from . import _geron as _core

from ._richresult import RichResult

__all__ = ["mcdrop", "geron_mc_dropout"]

_METHOD = "Monte Carlo dropout predictive average"


def mcdrop(logits):
    """Monte Carlo dropout predictive average.

    Average the softmax over repeated stochastic forward passes.

    The listing on p. 410 is ``softmax(logits)`` over a batch repeated
    T times, then ``.mean(dim=1)`` across the T passes.  ``logits`` here
    is that same three-level structure -- n instances by T passes by k
    classes -- supplied by the caller, so the dropout masks (which the
    book seeds with ``torch.manual_seed(42)``) live outside this
    function and both language arms see identical numbers.

    Parameters
    ----------
    logits : as documented for the shelf core
        See ``morie.fn._geron.mcdrop``.

    Returns
    -------
    result : RichResult
        Payload keys: meanmaxprob, meanmaxsd, meanentropy.

    References
    ----------
    Geron, A. (2026). Hands-On Machine Learning with Scikit-Learn and PyTorch. O'Reilly, ch. 11 p. 410
    """
    res = _core.mcdrop(logits=logits)
    return RichResult(
        title=_METHOD,
        summary_lines=[("meanmaxprob", res["meanmaxprob"]), ("meanmaxsd", res["meanmaxsd"]), ("meanentropy", res["meanentropy"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
geron_mc_dropout = mcdrop


def cheatsheet():
    return "mcdrop: Monte Carlo dropout predictive average"
