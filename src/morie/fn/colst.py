# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Collider structures in a DAG.

Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition), ch. 5 p. 82
"""

from . import _molak as _core

from ._richresult import RichResult

__all__ = ["collider", "collider_structure"]

_METHOD = "Collider structures in a DAG"


def collider(dag, triple=None):
    """Collider structures in a DAG.

    Collider structures (immoralities, v-structures), ch. 5 p. 82.

    With ``triple=(a, c, b)`` the return also says whether that one
    triple is a collider at ``c``.

    Parameters
    ----------
    dag : as documented for the shelf core
        See ``morie.fn._molak.collider``.
    triple : as documented for the shelf core
        See ``morie.fn._molak.collider``.

    Returns
    -------
    result : RichResult
        Payload keys: ncolliders, iscollider, nedges.

    References
    ----------
    Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition), ch. 5 p. 82
    """
    res = _core.collider(dag=dag, triple=triple)
    return RichResult(
        title=_METHOD,
        summary_lines=[("ncolliders", res["ncolliders"]), ("iscollider", res["iscollider"]), ("nedges", res["nedges"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
collider_structure = collider


def cheatsheet():
    return "collider: Collider structures in a DAG"
