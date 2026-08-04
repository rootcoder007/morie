# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Posterior over the mean at known noise level.

MacKay (2003) eq. (24.9)-(24.11), p. 320
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["mupostsg", "information_theory_mackay_chapter_24_equation_9"]

_METHOD = "Posterior over the mean at known noise level"


def mupostsg(xbar, n, sigma):
    """Posterior over the mean at known noise level.

    (24.9)-(24.11) p.320 -- P(mu | data, sigma) = Normal(xbar, sigma^2/n).

    Parameters
    ----------
    xbar : as documented for the shelf core
        See ``morie.fn._itila.mupostsg``.
    n : as documented for the shelf core
        See ``morie.fn._itila.mupostsg``.
    sigma : as documented for the shelf core
        See ``morie.fn._itila.mupostsg``.

    Returns
    -------
    result : RichResult
        Payload keys: mean, var, se.

    References
    ----------
    MacKay (2003) eq. (24.9)-(24.11), p. 320
    """
    res = _core.mupostsg(xbar=xbar, n=n, sigma=sigma)
    return RichResult(
        title=_METHOD,
        summary_lines=[("mean", res["mean"]), ("var", res["var"]), ("se", res["se"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_24_equation_9 = mupostsg


def cheatsheet():
    return "mupostsg: Posterior over the mean at known noise level -- MacKay (2003) eq. (24.9)-(24.11), p. 320"
