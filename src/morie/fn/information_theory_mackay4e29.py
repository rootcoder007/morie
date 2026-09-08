# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Typical-set membership test.

MacKay (2003) eq. (4.29), p. 80
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["typset", "information_theory_mackay_chapter_4_equation_29"]

_METHOD = "Typical-set membership test"


def typset(p, n, h, beta):
    """Typical-set membership test.

    (4.29) p.80 -- membership test for the typical set T_{N beta}.

    Parameters
    ----------
    p : as documented for the shelf core
        See ``morie.fn._itila.typset``.
    n : as documented for the shelf core
        See ``morie.fn._itila.typset``.
    h : as documented for the shelf core
        See ``morie.fn._itila.typset``.
    beta : as documented for the shelf core
        See ``morie.fn._itila.typset``.

    Returns
    -------
    result : RichResult
        Payload keys: info, rate, deviation.

    References
    ----------
    MacKay (2003) eq. (4.29), p. 80
    """
    res = _core.typset(p=p, n=n, h=h, beta=beta)
    return RichResult(
        title=_METHOD,
        summary_lines=[("info", res["info"]), ("rate", res["rate"]), ("deviation", res["deviation"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_4_equation_29 = typset


def cheatsheet():
    return "typset: Typical-set membership test -- MacKay (2003) eq. (4.29), p. 80"
