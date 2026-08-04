# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SUTVA: no interference and one version of treatment.

Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition), ch. 7 p. 164
"""

from . import _molak as _core

from ._richresult import RichResult

__all__ = ["sutvachk", "sutva_assumption"]

_METHOD = "SUTVA: no interference and one version of treatment"


def sutvachk(interference, versions=1, tol=0.0):
    """SUTVA: no interference and one version of treatment.

    SUTVA: no interference between units, one version of treatment.

    The printed statement (ch. 7 p. 164) is that "the fact that one
    unit receives treatment does not influence any other units".
    ``interference`` is a square matrix whose off-diagonal entry (i, j)
    is how much unit i's treatment moves unit j's outcome; SUTVA holds
    when every off-diagonal entry is at most ``tol`` in magnitude and
    there is a single version of the treatment.

    Parameters
    ----------
    interference : as documented for the shelf core
        See ``morie.fn._molak.sutvachk``.
    versions : as documented for the shelf core
        See ``morie.fn._molak.sutvachk``.
    tol : as documented for the shelf core
        See ``morie.fn._molak.sutvachk``.

    Returns
    -------
    result : RichResult
        Payload keys: maxinterference, holds, n.

    References
    ----------
    Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition), ch. 7 p. 164
    """
    res = _core.sutvachk(interference=interference, versions=versions, tol=tol)
    return RichResult(
        title=_METHOD,
        summary_lines=[("maxinterference", res["maxinterference"]), ("holds", res["holds"]), ("n", res["n"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
sutva_assumption = sutvachk


def cheatsheet():
    return "sutvachk: SUTVA: no interference and one version of treatment"
