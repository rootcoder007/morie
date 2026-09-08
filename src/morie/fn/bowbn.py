# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bow (confounded parent-child pair) test.

Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition) -- NOT LOCATED in the corpus copy
"""

from . import _molak as _core

from ._richresult import RichResult

__all__ = ["bowarc", "bow_ban_theorem"]

_METHOD = "Bow (confounded parent-child pair) test"


def bowarc(dag, bidirected, x, y):
    """Bow (confounded parent-child pair) test.

    Whether the pair (x, y) forms a bow.

    NOT LOCATED IN THE EXTRACTED TEXT of the corpus copy of Molak,
    which contains no discussion of bows or bow-free graphs.  The
    definition and the theorem are therefore taken from the primary
    source and quoted from it:

    "A bow-arc is a pair of variables, one of which is a direct
    function of the other, whose error terms are correlated."

    "Theorem 4. (Brito and Pearl, 2002b) (Bow-free Rule) Every
    acyclic model whose path diagram lacks bow-arcs is identified."

    -- Chen, B. and Pearl, J. (2015), *Graphical Tools for Linear
    Structural Equation Modeling*, UCLA Cognitive Systems Laboratory
    Technical Report R-432, p. 15, citing Brito, C. and Pearl, J.
    (2002), "A new identification condition for recursive models with
    correlated errors", Structural Equation Modeling 9(4):459-474.

    Footnote 17 of R-432 gives the equivalent phrasing used for
    ``bowfree`` here: "a bow-free model is a model where error terms of
    every parent-child pair are not correlated".

    ``identified`` reports Theorem 4 itself: acyclic AND bow-free.  The
    theorem addresses whole-model identifiability, not the
    identifiability of individual coefficients in unidentified models,
    so nothing stronger is claimed.

    Parameters
    ----------
    dag : as documented for the shelf core
        See ``morie.fn._molak.bowarc``.
    bidirected : as documented for the shelf core
        See ``morie.fn._molak.bowarc``.
    x : as documented for the shelf core
        See ``morie.fn._molak.bowarc``.
    y : as documented for the shelf core
        See ``morie.fn._molak.bowarc``.

    Returns
    -------
    result : RichResult
        Payload keys: isbow, direct, nbows.

    References
    ----------
    Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition) -- NOT LOCATED in the corpus copy
    """
    res = _core.bowarc(dag=dag, bidirected=bidirected, x=x, y=y)
    return RichResult(
        title=_METHOD,
        summary_lines=[("isbow", res["isbow"]), ("direct", res["direct"]), ("nbows", res["nbows"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
bow_ban_theorem = bowarc


def cheatsheet():
    return "bowarc: Bow (confounded parent-child pair) test"
