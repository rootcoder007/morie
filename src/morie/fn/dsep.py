# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""d-separation of two nodes given a conditioning set.

Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition), ch. 6
"""

from . import _molak as _core

from ._richresult import RichResult

__all__ = ["dseptest", "d_separation"]

_METHOD = "d-separation of two nodes given a conditioning set"


def dseptest(dag, x, y, z=()):
    """d-separation of two nodes given a conditioning set.

    d-separation of x and y given z, ch. 6.

    Reuses ``morie.fn._dsep.d_separated``; the extra counts come from
    the same path enumeration so callers can see WHY the answer came
    out the way it did.

    Parameters
    ----------
    dag : as documented for the shelf core
        See ``morie.fn._molak.dseptest``.
    x : as documented for the shelf core
        See ``morie.fn._molak.dseptest``.
    y : as documented for the shelf core
        See ``morie.fn._molak.dseptest``.
    z : as documented for the shelf core
        See ``morie.fn._molak.dseptest``.

    Returns
    -------
    result : RichResult
        Payload keys: dseparated, npaths, nnodes.

    References
    ----------
    Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition), ch. 6
    """
    res = _core.dseptest(dag=dag, x=x, y=y, z=z)
    return RichResult(
        title=_METHOD,
        summary_lines=[("dseparated", res["dseparated"]), ("npaths", res["npaths"]), ("nnodes", res["nnodes"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
d_separation = dseptest


def cheatsheet():
    return "dseptest: d-separation of two nodes given a conditioning set"
