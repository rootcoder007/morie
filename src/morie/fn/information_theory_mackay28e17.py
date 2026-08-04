# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Two-part minimum-description-length message length.

MacKay (2003) eq. (28.16)-(28.17), p. 352
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["mdlpost", "information_theory_mackay_chapter_28_equation_17"]

_METHOD = "Two-part minimum-description-length message length"


def mdlpost(ph, pdh, deltad=1.0):
    """Two-part minimum-description-length message length.

    (28.16)-(28.17) p.352 -- two-part MDL message length, in bits.

    Parameters
    ----------
    ph : as documented for the shelf core
        See ``morie.fn._itila.mdlpost``.
    pdh : as documented for the shelf core
        See ``morie.fn._itila.mdlpost``.
    deltad : as documented for the shelf core
        See ``morie.fn._itila.mdlpost``.

    Returns
    -------
    result : RichResult
        Payload keys: total, model, data.

    References
    ----------
    MacKay (2003) eq. (28.16)-(28.17), p. 352
    """
    res = _core.mdlpost(ph=ph, pdh=pdh, deltad=deltad)
    return RichResult(
        title=_METHOD,
        summary_lines=[("total", res["total"]), ("model", res["model"]), ("data", res["data"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_28_equation_17 = mdlpost


def cheatsheet():
    return "mdlpost: Two-part minimum-description-length message length -- MacKay (2003) eq. (28.16)-(28.17), p. 352"
