# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mean-fitness growth rate under recombination.

MacKay (2003) eq. (19.13), p. 273
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["sexdfdt", "information_theory_mackay_chapter_19_equation_13"]

_METHOD = "Mean-fitness growth rate under recombination"


def sexdfdt(f, g, eta=None):
    """Mean-fitness growth rate under recombination.

    (19.13) p.273 -- dF/dt = eta sqrt(f (1 - f) G) under sexual mixing.

    Parameters
    ----------
    f : as documented for the shelf core
        See ``morie.fn._itila.sexdfdt``.
    g : as documented for the shelf core
        See ``morie.fn._itila.sexdfdt``.
    eta : as documented for the shelf core
        See ``morie.fn._itila.sexdfdt``.

    Returns
    -------
    result : RichResult
        Payload keys: dfbardt, eta.

    References
    ----------
    MacKay (2003) eq. (19.13), p. 273
    """
    res = _core.sexdfdt(f=f, g=g, eta=eta)
    return RichResult(
        title=_METHOD,
        summary_lines=[("dfbardt", res["dfbardt"]), ("eta", res["eta"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_19_equation_13 = sexdfdt


def cheatsheet():
    return "sexdfdt: Mean-fitness growth rate under recombination -- MacKay (2003) eq. (19.13), p. 273"
