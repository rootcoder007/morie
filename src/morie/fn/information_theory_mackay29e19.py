# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Uniform draws needed to hit the typical set once.

MacKay (2003) eq. (29.19), p. 366
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["rminsamp", "information_theory_mackay_chapter_29_equation_19"]

_METHOD = "Uniform draws needed to hit the typical set once"


def rminsamp(n, h):
    """Uniform draws needed to hit the typical set once.

    (29.19) p.366 -- uniform draws needed to hit the typical set once.

    Parameters
    ----------
    n : as documented for the shelf core
        See ``morie.fn._itila.rminsamp``.
    h : as documented for the shelf core
        See ``morie.fn._itila.rminsamp``.

    Returns
    -------
    result : RichResult
        Payload keys: log2rmin, log10rmin.

    References
    ----------
    MacKay (2003) eq. (29.19), p. 366
    """
    res = _core.rminsamp(n=n, h=h)
    return RichResult(
        title=_METHOD,
        summary_lines=[("log2rmin", res["log2rmin"]), ("log10rmin", res["log10rmin"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_29_equation_19 = rminsamp


def cheatsheet():
    return "rminsamp: Uniform draws needed to hit the typical set once -- MacKay (2003) eq. (29.19), p. 366"
