# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Block error probability of the RN repetition code.

MacKay (2003) eq. (1.42)-(1.43), p. 17
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["repcpb", "information_theory_mackay_chapter_1_equation_43"]

_METHOD = "Block error probability of the RN repetition code"


def repcpb(n, f):
    """Block error probability of the RN repetition code.

    (1.42)-(1.43) p.17 -- block error probability of RN, odd N.

    Parameters
    ----------
    n : as documented for the shelf core
        See ``morie.fn._itila.repcpb``.
    f : as documented for the shelf core
        See ``morie.fn._itila.repcpb``.

    Returns
    -------
    result : RichResult
        Payload keys: leading, approx1, approx2.

    References
    ----------
    MacKay (2003) eq. (1.42)-(1.43), p. 17
    """
    res = _core.repcpb(n=n, f=f)
    return RichResult(
        title=_METHOD,
        summary_lines=[("leading", res["leading"]), ("approx1", res["approx1"]), ("approx2", res["approx2"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_1_equation_43 = repcpb


def cheatsheet():
    return "repcpb: Block error probability of the RN repetition code -- MacKay (2003) eq. (1.42)-(1.43), p. 17"
