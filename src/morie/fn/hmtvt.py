# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Three-way train/validation/test split by identifier hash.

Geron, A. (2026). Hands-On Machine Learning with Scikit-Learn and PyTorch. O'Reilly, ch. 2 p. 61 (hash generalization is ours, not the book's)
"""

from . import _geron as _core

from ._richresult import RichResult

__all__ = ["tvtsplit", "geron_train_val_test_split"]

_METHOD = "Three-way train/validation/test split by identifier hash"


def tvtsplit(ids, valratio=0.2, testratio=0.2):
    """Three-way train/validation/test split by identifier hash.

    Three-way train/validation/test split by identifier hash.

    p. 61 obtains a three-way split by calling the splitter twice; the
    hash generalization used here -- one CRC-32 per identifier, mapped
    to [0, 1) and cut at ``testratio`` and ``testratio + valratio`` --
    is OURS, not the book's, and is used because it stays deterministic
    across both language arms.  The p. 58 hash rule itself is the
    book's.

    Parameters
    ----------
    ids : as documented for the shelf core
        See ``morie.fn._geron.tvtsplit``.
    valratio : as documented for the shelf core
        See ``morie.fn._geron.tvtsplit``.
    testratio : as documented for the shelf core
        See ``morie.fn._geron.tvtsplit``.

    Returns
    -------
    result : RichResult
        Payload keys: ntrain, nval, ntest.

    References
    ----------
    Geron, A. (2026). Hands-On Machine Learning with Scikit-Learn and PyTorch. O'Reilly, ch. 2 p. 61 (hash generalization is ours, not the book's)
    """
    res = _core.tvtsplit(ids=ids, valratio=valratio, testratio=testratio)
    return RichResult(
        title=_METHOD,
        summary_lines=[("ntrain", res["ntrain"]), ("nval", res["nval"]), ("ntest", res["ntest"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
geron_train_val_test_split = tvtsplit


def cheatsheet():
    return "tvtsplit: Three-way train/validation/test split by identifier hash"
