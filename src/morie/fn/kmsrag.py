# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Self-RAG: reflection tokens that decide to retrieve, grade
relevance and check support."""

import re

from ._richresult import RichResult

__all__ = ["kamath_self_rag", "REFLECTION_TOKENS"]

# The reflection vocabulary, grouped by the decision each pair makes.
REFLECTION_TOKENS = {
    "retrieve": ("[Retrieve]", "[No Retrieve]"),
    "relevance": ("[Relevant]", "[Irrelevant]"),
    "support": ("[Supported]", "[Partially Supported]", "[No Support]"),
    "utility": ("[Utility:1]", "[Utility:2]", "[Utility:3]",
                "[Utility:4]", "[Utility:5]"),
}
_ALL = {t: g for g, toks in REFLECTION_TOKENS.items() for t in toks}


def kamath_self_rag(context, reflection_model, question=None):
    """The model emits tokens from
    {[Retrieve], [No Retrieve], [Relevant], [Irrelevant], [Supported],
    ...} and those tokens drive the pipeline.

    Orchestration with a CLOSED vocabulary: a token outside the set is
    an error, and two contradictory tokens from the same group (both
    [Relevant] and [Irrelevant]) are an error too. That check is the
    whole value of this wrapper -- an unrecognised reflection token
    silently ignored turns Self-RAG back into ordinary RAG.

    Reference: the worklist cites Kamath, Keenan, Somers and Sorenson
    (2024), *Large Language Models: A Deep Dive*, Springer, Ch 7,
    Self-RAG; that section is not in the 2024 PDF, so the token set
    and its semantics follow the spec line (Asai et al. 2023).

    Examples
    --------
    >>> out = kamath_self_rag(["doc"],
    ...     lambda c, q: ["[Retrieve]", "[Relevant]", "[Supported]"])
    >>> out["retrieve"], out["relevant"], out["supported"]
    (True, True, True)
    >>> out["estimate"]
    3
    >>> skip = kamath_self_rag(["doc"], lambda c, q: ["[No Retrieve]"])
    >>> skip["retrieve"], skip["relevant"]
    (False, None)
    """
    if not callable(reflection_model):
        raise ValueError(
            "reflection_model must be callable (context, question) -> "
            "reflection tokens.")
    toks = reflection_model(context, question)
    if isinstance(toks, str):
        # Split on the bracket, not on whitespace: "[No Retrieve]" is ONE
        # token and whitespace-splitting silently shreds it into two
        # unrecognised ones.
        toks = re.findall(r"\[[^\]]*\]", toks)
    toks = list(toks)
    if not toks:
        raise ValueError(
            "the reflection model emitted no tokens; Self-RAG is "
            "defined by those tokens, so there is no decision to make.")
    groups = {}
    for t in toks:
        g = _ALL.get(t)
        if g is None:
            raise ValueError(
                f"{t!r} is not a Self-RAG reflection token; the "
                f"vocabulary is {sorted(_ALL)}.")
        if g in groups and groups[g] != t:
            raise ValueError(
                f"contradictory {g} tokens: {groups[g]!r} and {t!r}.")
        groups[g] = t

    def _flag(group, positive):
        t = groups.get(group)
        return None if t is None else (t == positive)

    utility = None
    if "utility" in groups:
        utility = int(groups["utility"].split(":")[1].rstrip("]"))
    return RichResult(payload={
        "tokens": toks, "by_group": groups,
        "retrieve": _flag("retrieve", "[Retrieve]"),
        "relevant": _flag("relevance", "[Relevant]"),
        "supported": (None if "support" not in groups
                      else groups["support"] == "[Supported]"),
        "support_level": groups.get("support"),
        "utility": utility,
        "estimate": len(groups), "n": len(toks),
        "method": "Self-RAG reflection-token decisions"})


def cheatsheet():
    return "kmsrag: closed reflection vocabulary; unknown or contradictory tokens refused"
