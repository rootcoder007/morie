# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rule of succession.

MacKay (2003) eq. (3.16), p. 52
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["sucrule", "information_theory_mackay_chapter_3_equation_16"]

_METHOD = "Rule of succession"


def sucrule(fa, fb):
    """Rule of succession.

    (3.16) p.52 -- the rule of succession, (Fa + 1)/(Fa + Fb + 2).

    Parameters
    ----------
    fa : as documented for the shelf core
        See ``morie.fn._itila.sucrule``.
    fb : as documented for the shelf core
        See ``morie.fn._itila.sucrule``.

    Returns
    -------
    result : RichResult
        Payload keys: p, pnot, mle.

    References
    ----------
    MacKay (2003) eq. (3.16), p. 52
    """
    res = _core.sucrule(fa=fa, fb=fb)
    return RichResult(
        title=_METHOD,
        summary_lines=[("p", res["p"]), ("pnot", res["pnot"]), ("mle", res["mle"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_3_equation_16 = sucrule


def cheatsheet():
    return "sucrule: Rule of succession -- MacKay (2003) eq. (3.16), p. 52"
