# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Message length and its probability, in bits.

MacKay (2003) eq. (28.15), p. 352
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["msglen", "information_theory_mackay_chapter_28_equation_15"]

_METHOD = "Message length and its probability, in bits"


def msglen(p=None, length=None):
    """Message length and its probability, in bits.

    (28.15) p.352 -- P(x) = 2^-L(x), L(x) = -log2 P(x).

    Parameters
    ----------
    p : as documented for the shelf core
        See ``morie.fn._itila.msglen``.
    length : as documented for the shelf core
        See ``morie.fn._itila.msglen``.

    Returns
    -------
    result : RichResult
        Payload keys: length, p, nats.

    References
    ----------
    MacKay (2003) eq. (28.15), p. 352
    """
    res = _core.msglen(p=p, length=length)
    return RichResult(
        title=_METHOD,
        summary_lines=[("length", res["length"]), ("p", res["p"]), ("nats", res["nats"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_28_equation_15 = msglen


def cheatsheet():
    return "msglen: Message length and its probability, in bits -- MacKay (2003) eq. (28.15), p. 352"
