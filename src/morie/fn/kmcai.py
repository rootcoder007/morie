# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 5: the Constitutional AI critique-and-revise loop."""

from ._richresult import RichResult

__all__ = ["kamath_constitutional_ai_loop"]


def kamath_constitutional_ai_loop(initial_response, constitution, model):
    r"""For each principle: critique the response, then revise it.

    ``model`` is the caller's LLM under a fixed contract:
    ``model(stage, principle, response, critique)`` where ``stage`` is
    ``"critique"`` (``critique`` is then ``None``) or ``"revise"``.
    Each principle sees the CURRENT response, not the original -- the
    revisions compose, which is the point of the loop -- and every
    step's text is kept in ``history`` so the chain is auditable.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 5, Constitutional AI;
    Bai et al. (2022).

    Examples
    --------
    >>> def m(stage, principle, response, critique):
    ...     return "too blunt" if stage == "critique" else response + "!"
    >>> out = kamath_constitutional_ai_loop("no", ["be kind", "be brief"], m)
    >>> out["revised_response"], out["n_revisions"]
    ('no!!', 2)
    """
    if not callable(model):
        raise ValueError("model must be callable model(stage, "
                         "principle, response, critique).")
    principles = list(constitution)
    if len(principles) == 0:
        raise ValueError("the constitution is empty; there is nothing "
                         "to critique against.")
    y = initial_response
    history = []
    for p in principles:
        crit = model("critique", p, y, None)
        if crit is None:
            raise ValueError(f"the model returned no critique for "
                             f"principle {p!r}.")
        rev = model("revise", p, y, crit)
        if rev is None:
            raise ValueError(f"the model returned no revision for "
                             f"principle {p!r}.")
        history.append({"principle": p, "critique": crit,
                        "response_before": y, "response_after": rev})
        y = rev
    return RichResult(payload={
        "estimate": y, "revised_response": y,
        "initial_response": initial_response, "history": history,
        "n_revisions": len(principles), "n": len(principles),
        "method": "Constitutional AI critique-revise loop "
                  "(Kamath Ch 5)"})


def cheatsheet():
    return "kmcai: critique then revise once per constitutional principle"
