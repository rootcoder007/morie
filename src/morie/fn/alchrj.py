# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chosen/rejected preference records (Alammar Ch 12)."""

from ._richresult import RichResult

__all__ = ["alammar_chosen_rejected_template"]


def alammar_chosen_rejected_template(prompts, chosen, rejected):
    """One record per prompt: {prompt, chosen y_w, rejected y_l}.

    A record whose chosen and rejected are IDENTICAL is refused --
    it encodes no preference and silently dilutes a DPO/RM dataset.

    Examples
    --------
    >>> out = alammar_chosen_rejected_template(["p"], ["good"], ["bad"])
    >>> out["records"][0]["chosen"]
    'good'
    """
    P = [str(p) for p in prompts]
    C = [str(c) for c in chosen]
    R = [str(r) for r in rejected]
    if not (len(P) == len(C) == len(R)):
        raise ValueError("prompts, chosen and rejected must align.")
    if not P:
        raise ValueError("no records supplied.")
    for i, (c, r) in enumerate(zip(C, R)):
        if c == r:
            raise ValueError(
                f"record {i} has identical chosen and rejected; it "
                "encodes no preference.")
    recs = [{"prompt": p, "chosen": c, "rejected": r}
            for p, c, r in zip(P, C, R)]
    return RichResult(payload={
        "records": recs, "estimate": float(len(recs)), "n": len(recs),
        "method": "Preference pair records (Alammar Ch 12)"})


def cheatsheet():
    return "alchrj: {prompt, y_w, y_l} records, ties refused"
