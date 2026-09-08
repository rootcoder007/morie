# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The do-operator as graph surgery.

Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition), ch. 7 p. 154
"""

from . import _molak as _core

from ._richresult import RichResult

__all__ = ["dointerv", "do_operator"]

_METHOD = "The do-operator as graph surgery"


def dointerv(dag, x):
    """The do-operator as graph surgery.

    The do-operator as graph surgery (modularity, ch. 7 p. 154).

    ``do(X = x)`` deletes every edge into X and leaves every other
    structural equation untouched.

    Parameters
    ----------
    dag : as documented for the shelf core
        See ``morie.fn._molak.dointerv``.
    x : as documented for the shelf core
        See ``morie.fn._molak.dointerv``.

    Returns
    -------
    result : RichResult
        Payload keys: nremoved, nkept, nnodes.

    References
    ----------
    Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition), ch. 7 p. 154
    """
    res = _core.dointerv(dag=dag, x=x)
    return RichResult(
        title=_METHOD,
        summary_lines=[("nremoved", res["nremoved"]), ("nkept", res["nkept"]), ("nnodes", res["nnodes"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
do_operator = dointerv


def cheatsheet():
    return "dointerv: The do-operator as graph surgery"
