# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Posterior model ratio as a product of parameter penalties.

MacKay (2003) eq. (28.13)-(28.14), p. 351
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["evratio", "information_theory_mackay_chapter_28_equation_13"]

_METHOD = "Posterior model ratio as a product of parameter penalties"


def evratio(factors):
    """Posterior model ratio as a product of parameter penalties.

    (28.13)-(28.14) p.351 -- posterior ratio as a product of penalties.

    Parameters
    ----------
    factors : as documented for the shelf core
        See ``morie.fn._itila.evratio``.

    Returns
    -------
    result : RichResult
        Payload keys: ratio, product, logratio.

    References
    ----------
    MacKay (2003) eq. (28.13)-(28.14), p. 351
    """
    res = _core.evratio(factors=factors)
    return RichResult(
        title=_METHOD,
        summary_lines=[("ratio", res["ratio"]), ("product", res["product"]), ("logratio", res["logratio"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_28_equation_13 = evratio


def cheatsheet():
    return "evratio: Posterior model ratio as a product of parameter penalties -- MacKay (2003) eq. (28.13)-(28.14), p. 351"
