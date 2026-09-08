# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Positivity assumption over covariate strata.

Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition), ch. 7 p. 157
"""

from . import _molak as _core

from ._richresult import RichResult

__all__ = ["poschk", "positivity_assumption"]

_METHOD = "Positivity assumption over covariate strata"


def poschk(treat, stratum, tol=0.0):
    """Positivity assumption over covariate strata.

    Positivity: every treatment value has positive probability in
    every covariate stratum, ch. 7 p. 157.

    Returns the smallest stratum-conditional treatment probability over
    all (stratum, treatment level) cells, so the caller can see how
    close to a violation the data sit rather than only that it passed.

    Parameters
    ----------
    treat : as documented for the shelf core
        See ``morie.fn._molak.poschk``.
    stratum : as documented for the shelf core
        See ``morie.fn._molak.poschk``.
    tol : as documented for the shelf core
        See ``morie.fn._molak.poschk``.

    Returns
    -------
    result : RichResult
        Payload keys: minprob, holds, ncells.

    References
    ----------
    Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition), ch. 7 p. 157
    """
    res = _core.poschk(treat=treat, stratum=stratum, tol=tol)
    return RichResult(
        title=_METHOD,
        summary_lines=[("minprob", res["minprob"]), ("holds", res["holds"]), ("ncells", res["ncells"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
positivity_assumption = poschk


def cheatsheet():
    return "poschk: Positivity assumption over covariate strata"
