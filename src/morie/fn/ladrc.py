# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One rung of the Ladder of Causation.

Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition), Table 2.1, ch. 2 p. 15
"""

from . import _molak as _core

from ._richresult import RichResult

__all__ = ["causrung", "ladder_of_causation"]

_METHOD = "One rung of the Ladder of Causation"


def causrung(rung):
    """One rung of the Ladder of Causation.

    One rung of the Ladder of Causation, Table 2.1, ch. 2 p. 15.

    The three rungs and their actions and questions are transcribed
    from the printed table.  ``needsgraph`` and ``needsscm`` follow the
    book's own account of what each rung requires: rung 1 needs only a
    joint distribution, rung 2 needs a causal graph, rung 3 needs a
    full structural causal model.

    Parameters
    ----------
    rung : as documented for the shelf core
        See ``morie.fn._molak.causrung``.

    Returns
    -------
    result : RichResult
        Payload keys: level, needsgraph, needsscm.

    References
    ----------
    Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition), Table 2.1, ch. 2 p. 15
    """
    res = _core.causrung(rung=rung)
    return RichResult(
        title=_METHOD,
        summary_lines=[("level", res["level"]), ("needsgraph", res["needsgraph"]), ("needsscm", res["needsscm"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
ladder_of_causation = causrung


def cheatsheet():
    return "causrung: One rung of the Ladder of Causation"
