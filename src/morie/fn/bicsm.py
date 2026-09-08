# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gaussian BIC score of a DAG.

Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition), ch. 13 p. 348
"""

from . import _molak as _core

from ._richresult import RichResult

__all__ = ["bicdag", "bic_score_dag"]

_METHOD = "Gaussian BIC score of a DAG"


def bicdag(data, dag, names=None):
    """Gaussian BIC score of a DAG.

    Gaussian BIC score of a DAG.

    The corpus copy only NAMES the Bayesian Information Criterion as a
    gCastle GES scoring option (ch. 13, p. 348, citing Chickering
    2003); no BIC formula is printed there.  The formula used here is
    the standard Gaussian one, stated explicitly so nothing is
    attributed to the book that the book does not say:

    score(G) = sum_j [ -n/2 (log(2 pi s2_j) + 1) ] - (log n / 2) k

    with ``s2_j`` the residual variance of the OLS regression of node j
    on its parents and ``k`` the free-parameter count (one intercept,
    one slope per parent, one variance per node).

    Parameters
    ----------
    data : as documented for the shelf core
        See ``morie.fn._molak.bicdag``.
    dag : as documented for the shelf core
        See ``morie.fn._molak.bicdag``.
    names : as documented for the shelf core
        See ``morie.fn._molak.bicdag``.

    Returns
    -------
    result : RichResult
        Payload keys: score, loglik, k, penalty.

    References
    ----------
    Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition), ch. 13 p. 348
    """
    res = _core.bicdag(data=data, dag=dag, names=names)
    return RichResult(
        title=_METHOD,
        summary_lines=[("score", res["score"]), ("loglik", res["loglik"]), ("k", res["k"]), ("penalty", res["penalty"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
bic_score_dag = bicdag


def cheatsheet():
    return "bicdag: Gaussian BIC score of a DAG"
