# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Total evidence over a set of models by the sum rule.

MacKay (2003) eq. (3.19), p. 53
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["evidmix", "information_theory_mackay_chapter_3_equation_19"]

_METHOD = "Total evidence over a set of models by the sum rule"


def evidmix(evidences, priors):
    """Total evidence over a set of models by the sum rule.

    (3.19) p.53 -- total evidence P(s | F) by the sum rule.

    Parameters
    ----------
    evidences : as documented for the shelf core
        See ``morie.fn._itila.evidmix``.
    priors : as documented for the shelf core
        See ``morie.fn._itila.evidmix``.

    Returns
    -------
    result : RichResult
        Payload keys: evidence.

    References
    ----------
    MacKay (2003) eq. (3.19), p. 53
    """
    res = _core.evidmix(evidences=evidences, priors=priors)
    return RichResult(
        title=_METHOD,
        summary_lines=[("evidence", res["evidence"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_3_equation_19 = evidmix


def cheatsheet():
    return "evidmix: Total evidence over a set of models by the sum rule -- MacKay (2003) eq. (3.19), p. 53"
