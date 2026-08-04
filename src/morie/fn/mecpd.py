# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Markov equivalence of two DAGs.

Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition), ch. 5 p. 85
"""

from . import _molak as _core

from ._richresult import RichResult

__all__ = ["mectest", "markov_equivalence_class"]

_METHOD = "Markov equivalence of two DAGs"


def mectest(dag1, dag2):
    """Markov equivalence of two DAGs.

    Markov equivalence, ch. 5 p. 85 (Verma and Pearl, 1991).

    Two DAGs are Markov equivalent iff they share a skeleton and a set
    of colliders.

    Parameters
    ----------
    dag1 : as documented for the shelf core
        See ``morie.fn._molak.mectest``.
    dag2 : as documented for the shelf core
        See ``morie.fn._molak.mectest``.

    Returns
    -------
    result : RichResult
        Payload keys: equivalent, sameskeleton, nskeleton.

    References
    ----------
    Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition), ch. 5 p. 85
    """
    res = _core.mectest(dag1=dag1, dag2=dag2)
    return RichResult(
        title=_METHOD,
        summary_lines=[("equivalent", res["equivalent"]), ("sameskeleton", res["sameskeleton"]), ("nskeleton", res["nskeleton"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
markov_equivalence_class = mectest


def cheatsheet():
    return "mectest: Markov equivalence of two DAGs"
