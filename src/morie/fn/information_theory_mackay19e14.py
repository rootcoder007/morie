# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Closed-form fitness trajectory under recombination.

MacKay (2003) eq. (19.14), p. 273
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["sexfsol", "information_theory_mackay_chapter_19_equation_14"]

_METHOD = "Closed-form fitness trajectory under recombination"


def sexfsol(t, g, f0, eta=None, c=None):
    """Closed-form fitness trajectory under recombination.

    (19.14) p.273 -- f(t) = (1 + sin(eta (t + c)/sqrt(G)))/2.

    The book states ``c = asin(2 f(0) - 1)``, which does NOT satisfy its
    own f(0) = f_0 unless the sine argument is read as
    ``eta t/sqrt(G) + c``.  The default ``c`` here is the
    self-consistent one, ``sqrt(G)/eta * asin(2 f_0 - 1)``; the printed
    value is returned as ``cbook`` so the disagreement stays visible.

    Parameters
    ----------
    t : as documented for the shelf core
        See ``morie.fn._itila.sexfsol``.
    g : as documented for the shelf core
        See ``morie.fn._itila.sexfsol``.
    f0 : as documented for the shelf core
        See ``morie.fn._itila.sexfsol``.
    eta : as documented for the shelf core
        See ``morie.fn._itila.sexfsol``.
    c : as documented for the shelf core
        See ``morie.fn._itila.sexfsol``.

    Returns
    -------
    result : RichResult
        Payload keys: f, c, tperfect.

    References
    ----------
    MacKay (2003) eq. (19.14), p. 273
    """
    res = _core.sexfsol(t=t, g=g, f0=f0, eta=eta, c=c)
    return RichResult(
        title=_METHOD,
        summary_lines=[("f", res["f"]), ("c", res["c"]), ("tperfect", res["tperfect"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_19_equation_14 = sexfsol


def cheatsheet():
    return "sexfsol: Closed-form fitness trajectory under recombination -- MacKay (2003) eq. (19.14), p. 273"
