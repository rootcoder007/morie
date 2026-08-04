# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stable train/test split by identifier hash.

Geron, A. (2026). Hands-On Machine Learning with Scikit-Learn and PyTorch. O'Reilly, ch. 2 p. 58
"""

from . import _geron as _core

from ._richresult import RichResult

__all__ = ["ttsplit", "geron_train_test_split"]

_METHOD = "Stable train/test split by identifier hash"


def ttsplit(ids, testratio=0.2):
    """Stable train/test split by identifier hash.

    Stable train/test split by identifier hash, p. 58.

    The book's own stable alternative to a seeded shuffle: an instance
    goes to the test set when ``crc32(int64(id)) < testratio * 2**32``.
    It is a pure function of the identifiers, which is exactly why the
    book prefers it -- and why it survives translation to R unchanged.

    Parameters
    ----------
    ids : as documented for the shelf core
        See ``morie.fn._geron.ttsplit``.
    testratio : as documented for the shelf core
        See ``morie.fn._geron.ttsplit``.

    Returns
    -------
    result : RichResult
        Payload keys: ntest, ntrain, ratio.

    References
    ----------
    Geron, A. (2026). Hands-On Machine Learning with Scikit-Learn and PyTorch. O'Reilly, ch. 2 p. 58
    """
    res = _core.ttsplit(ids=ids, testratio=testratio)
    return RichResult(
        title=_METHOD,
        summary_lines=[("ntest", res["ntest"]), ("ntrain", res["ntrain"]), ("ratio", res["ratio"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
geron_train_test_split = ttsplit


def cheatsheet():
    return "ttsplit: Stable train/test split by identifier hash"
