# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Applicability of the three rules of do-calculus.

Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition), ch. 6 p. 119
"""

from . import _molak as _core

from ._richresult import RichResult

__all__ = ["docalc", "do_calculus_rules"]

_METHOD = "Applicability of the three rules of do-calculus"


def docalc(dag, y, z, x=(), w=()):
    """Applicability of the three rules of do-calculus.

    Which of the three rules of do-calculus applies, p. 119.

    Notation as printed: ``G`` with an overline on X drops every edge
    INTO X; with an underline on Z drops every edge OUT OF Z.

    * Rule 1 (ignore an observation): (Y indep Z | X, W) in G_Xbar
    * Rule 2 (treat an intervention as an observation):
    (Y indep Z | X, W) in G_Xbar,Zunder
    * Rule 3 (ignore an intervention): (Y indep Z | X, W) in
    G_Xbar,Zbar(W), where Z(W) is the subset of Z that are not
    ancestors of any W node in G_Xbar.

    Parameters
    ----------
    dag : as documented for the shelf core
        See ``morie.fn._molak.docalc``.
    y : as documented for the shelf core
        See ``morie.fn._molak.docalc``.
    z : as documented for the shelf core
        See ``morie.fn._molak.docalc``.
    x : as documented for the shelf core
        See ``morie.fn._molak.docalc``.
    w : as documented for the shelf core
        See ``morie.fn._molak.docalc``.

    Returns
    -------
    result : RichResult
        Payload keys: rule1, rule2, rule3, nrules.

    References
    ----------
    Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition), ch. 6 p. 119
    """
    res = _core.docalc(dag=dag, y=y, z=z, x=x, w=w)
    return RichResult(
        title=_METHOD,
        summary_lines=[("rule1", res["rule1"]), ("rule2", res["rule2"]), ("rule3", res["rule3"]), ("nrules", res["nrules"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
do_calculus_rules = docalc


def cheatsheet():
    return "docalc: Applicability of the three rules of do-calculus"
