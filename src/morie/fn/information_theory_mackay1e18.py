# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""R3 repetition-code bit posterior.

MacKay (2003) eq. (1.18), p. 9
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["r3post", "information_theory_mackay_chapter_1_equation_18"]

_METHOD = "R3 repetition-code bit posterior"


def r3post(r, f):
    """R3 repetition-code bit posterior.

    (1.18) p.9 -- posterior over the source bit of an R3 codeword.

    Parameters
    ----------
    r : as documented for the shelf core
        See ``morie.fn._itila.r3post``.
    f : as documented for the shelf core
        See ``morie.fn._itila.r3post``.

    Returns
    -------
    result : RichResult
        Payload keys: p1, p0, decoded.

    References
    ----------
    MacKay (2003) eq. (1.18), p. 9
    """
    res = _core.r3post(r=r, f=f)
    return RichResult(
        title=_METHOD,
        summary_lines=[("p1", res["p1"]), ("p0", res["p0"]), ("decoded", res["decoded"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_1_equation_18 = r3post


def cheatsheet():
    return "r3post: R3 repetition-code bit posterior -- MacKay (2003) eq. (1.18), p. 9"
