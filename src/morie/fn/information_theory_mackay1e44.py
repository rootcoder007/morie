# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Repetition-code blocklength reaching a target error rate.

MacKay (2003) eq. (1.44)-(1.45), p. 17
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["repcn", "information_theory_mackay_chapter_1_equation_44"]

_METHOD = "Repetition-code blocklength reaching a target error rate"


def repcn(pb, f, n0=68.0, iters=3):
    """Repetition-code blocklength reaching a target error rate.

    (1.44)-(1.45) p.17 -- blocklength N reaching a target pb.

    Fixed ``iters`` sweeps of the iteration printed under (1.44),
    started from the book value N-hat_1 = 68; there is no convergence
    test, so both language arms take identical steps.

    Parameters
    ----------
    pb : as documented for the shelf core
        See ``morie.fn._itila.repcn``.
    f : as documented for the shelf core
        See ``morie.fn._itila.repcn``.
    n0 : as documented for the shelf core
        See ``morie.fn._itila.repcn``.
    iters : as documented for the shelf core
        See ``morie.fn._itila.repcn``.

    Returns
    -------
    result : RichResult
        Payload keys: n, half, denom.

    References
    ----------
    MacKay (2003) eq. (1.44)-(1.45), p. 17
    """
    res = _core.repcn(pb=pb, f=f, n0=n0, iters=iters)
    return RichResult(
        title=_METHOD,
        summary_lines=[("n", res["n"]), ("half", res["half"]), ("denom", res["denom"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_1_equation_44 = repcn


def cheatsheet():
    return "repcn: Repetition-code blocklength reaching a target error rate -- MacKay (2003) eq. (1.44)-(1.45), p. 17"
