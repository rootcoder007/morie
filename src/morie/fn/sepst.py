# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Smallest separating set for two nodes.

Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition), ch. 6
"""

from . import _molak as _core

from ._richresult import RichResult

__all__ = ["sepset", "separation_set"]

_METHOD = "Smallest separating set for two nodes"


def sepset(dag, x, y, maxsize=3):
    """Smallest separating set for two nodes.

    Smallest set that d-separates x from y, searched in a fixed order.

    The corpus copy discusses conditioning sets that block paths (ch. 6)
    but prints no named "separating set" definition, so the search rule
    is stated here: candidate sets are drawn from the union of the
    adjacencies of x and y, taken in sorted order and in increasing
    size, and the FIRST separating set found is returned.  That order
    is deterministic, so both language arms return the same set.

    Parameters
    ----------
    dag : as documented for the shelf core
        See ``morie.fn._molak.sepset``.
    x : as documented for the shelf core
        See ``morie.fn._molak.sepset``.
    y : as documented for the shelf core
        See ``morie.fn._molak.sepset``.
    maxsize : as documented for the shelf core
        See ``morie.fn._molak.sepset``.

    Returns
    -------
    result : RichResult
        Payload keys: found, size, ntested.

    References
    ----------
    Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition), ch. 6
    """
    res = _core.sepset(dag=dag, x=x, y=y, maxsize=maxsize)
    return RichResult(
        title=_METHOD,
        summary_lines=[("found", res["found"]), ("size", res["size"]), ("ntested", res["ntested"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
separation_set = sepset


def cheatsheet():
    return "sepset: Smallest separating set for two nodes"
