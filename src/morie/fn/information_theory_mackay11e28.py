# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Posterior over the input of a Gaussian channel.

MacKay (2003) eq. (11.27)-(11.29), p. 182
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["gchpost", "information_theory_mackay_chapter_11_equation_28"]

_METHOD = "Posterior over the input of a Gaussian channel"


def gchpost(y, v, s2):
    """Posterior over the input of a Gaussian channel.

    (11.27)-(11.29) p.182 -- posterior over a Gaussian channel input.

    Parameters
    ----------
    y : as documented for the shelf core
        See ``morie.fn._itila.gchpost``.
    v : as documented for the shelf core
        See ``morie.fn._itila.gchpost``.
    s2 : as documented for the shelf core
        See ``morie.fn._itila.gchpost``.

    Returns
    -------
    result : RichResult
        Payload keys: mean, var, sd.

    References
    ----------
    MacKay (2003) eq. (11.27)-(11.29), p. 182
    """
    res = _core.gchpost(y=y, v=v, s2=s2)
    return RichResult(
        title=_METHOD,
        summary_lines=[("mean", res["mean"]), ("var", res["var"]), ("sd", res["sd"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_11_equation_28 = gchpost


def cheatsheet():
    return "gchpost: Posterior over the input of a Gaussian channel -- MacKay (2003) eq. (11.27)-(11.29), p. 182"
