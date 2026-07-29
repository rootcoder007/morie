# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Step-back prompting: ask the general question first, then the
specific one."""

from ._richresult import RichResult

__all__ = ["kamath_step_back_prompting"]


def kamath_step_back_prompting(query, model, retrieve=None, answer=None):
    """q_high = LLM(step_back(x)); answer conditioned on the context
    retrieved for BOTH q_high and x.

    The retrieved sets are unioned with the abstraction's documents
    first and duplicates removed while keeping order -- a document
    found by both queries must not be counted twice in the context
    window. If the model returns the original question unchanged it
    has not stepped back, and that is reported rather than passed off
    as an abstraction.

    Reference: the worklist cites Kamath, Keenan, Somers and Sorenson
    (2024), *Large Language Models: A Deep Dive*, Springer, Ch 7,
    step-back prompting; that section is not in the 2024 PDF, so the
    two-stage procedure follows the spec line (Zheng et al. 2023).

    Examples
    --------
    >>> docs = {"physics": ["d1", "d2"], "which force": ["d2", "d3"]}
    >>> out = kamath_step_back_prompting(
    ...     "which force", lambda q: "physics",
    ...     retrieve=lambda q: docs[q],
    ...     answer=lambda q, ctx: "gravity, per " + ",".join(ctx))
    >>> out["step_back_query"]
    'physics'
    >>> out["context"]
    ['d1', 'd2', 'd3']
    >>> out["answer"]
    'gravity, per d1,d2,d3'
    """
    if not callable(model):
        raise ValueError(
            "model must be callable query -> the stepped-back query.")
    q_high = model(query)
    if q_high is None:
        raise ValueError("the model returned no step-back query.")
    stepped = q_high != query

    ctx, seen = [], set()
    per_query = {}
    if retrieve is not None:
        if not callable(retrieve):
            raise ValueError("retrieve must be callable query -> documents.")
        for q in (q_high, query):
            docs = list(retrieve(q))
            per_query[q] = docs
            for d in docs:
                key = repr(d)
                if key not in seen:
                    seen.add(key)
                    ctx.append(d)
    payload = {
        "step_back_query": q_high, "query": query,
        "stepped_back": stepped,
        "context": ctx, "retrieved_by_query": per_query,
        "n_context": len(ctx),
        "estimate": len(ctx), "n": len(ctx),
        "method": "Step-back prompting (abstract query, then specific)"}
    if not stepped:
        payload["warning"] = (
            "the model returned the original question, so no "
            "abstraction was made.")
    if answer is not None:
        if not callable(answer):
            raise ValueError("answer must be callable (query, context) -> str.")
        payload["answer"] = answer(query, ctx)
    return RichResult(payload=payload)


def cheatsheet():
    return "kmstb: LLM abstracts the query, contexts unioned, order kept"
