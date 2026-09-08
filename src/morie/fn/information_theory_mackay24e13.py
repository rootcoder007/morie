# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Log evidence for the noise level, mean marginalized out.

MacKay (2003) eq. (24.13), p. 320
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["sigevid", "information_theory_mackay_chapter_24_equation_13"]

_METHOD = "Log evidence for the noise level, mean marginalized out"


def sigevid(s, n, sigma, sigmamu=1.0):
    """Log evidence for the noise level, mean marginalized out.

    (24.13) p.320 -- log evidence for sigma, mu marginalized out.

    Parameters
    ----------
    s : as documented for the shelf core
        See ``morie.fn._itila.sigevid``.
    n : as documented for the shelf core
        See ``morie.fn._itila.sigevid``.
    sigma : as documented for the shelf core
        See ``morie.fn._itila.sigevid``.
    sigmamu : as documented for the shelf core
        See ``morie.fn._itila.sigevid``.

    Returns
    -------
    result : RichResult
        Payload keys: logevidence, bestfit, logoccam.

    References
    ----------
    MacKay (2003) eq. (24.13), p. 320
    """
    res = _core.sigevid(s=s, n=n, sigma=sigma, sigmamu=sigmamu)
    return RichResult(
        title=_METHOD,
        summary_lines=[("logevidence", res["logevidence"]), ("bestfit", res["bestfit"]), ("logoccam", res["logoccam"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_24_equation_13 = sigevid


def cheatsheet():
    return "sigevid: Log evidence for the noise level, mean marginalized out -- MacKay (2003) eq. (24.13), p. 320"
