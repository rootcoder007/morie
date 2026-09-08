# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Faithfulness assumption for one triple.

Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition), ch. 5 p. 77
"""

from . import _molak as _core

from ._richresult import RichResult

__all__ = ["faithchk", "faithfulness_assumption"]

_METHOD = "Faithfulness assumption for one triple"


def faithchk(dag, x, y, z=(), indep=True):
    """Faithfulness assumption for one triple.

    Faithfulness for one triple, ch. 5 p. 77.

    The printed formulation is ``X indep_P Y | Z  =>  X indep_G Y | Z``:
    an independence in the DISTRIBUTION must be reflected in the GRAPH.
    ``indep`` is the observed distributional independence; the return
    says whether that implication survives for this triple, and
    separately whether the converse (the global Markov property) does.

    Parameters
    ----------
    dag : as documented for the shelf core
        See ``morie.fn._molak.faithchk``.
    x : as documented for the shelf core
        See ``morie.fn._molak.faithchk``.
    y : as documented for the shelf core
        See ``morie.fn._molak.faithchk``.
    z : as documented for the shelf core
        See ``morie.fn._molak.faithchk``.
    indep : as documented for the shelf core
        See ``morie.fn._molak.faithchk``.

    Returns
    -------
    result : RichResult
        Payload keys: dseparated, faithful, markov.

    References
    ----------
    Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition), ch. 5 p. 77
    """
    res = _core.faithchk(dag=dag, x=x, y=y, z=z, indep=indep)
    return RichResult(
        title=_METHOD,
        summary_lines=[("dseparated", res["dseparated"]), ("faithful", res["faithful"]), ("markov", res["markov"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
faithfulness_assumption = faithchk


def cheatsheet():
    return "faithchk: Faithfulness assumption for one triple"
