# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Instruction-tuning template with output-region loss mask
(Alammar Ch 11)."""

from ._richresult import RichResult

__all__ = ["alammar_instruction_data_template"]


def alammar_instruction_data_template(records, template=None):
    """text = fmt(instruction, input, output); the SFT loss is masked
    to the OUTPUT region, so its character span is returned per record
    -- training on the instruction tokens too is the classic silent
    bug this template exists to prevent.

    Examples
    --------
    >>> out = alammar_instruction_data_template(
    ...     [{"instruction": "add", "input": "2 2", "output": "4"}])
    >>> t = out["texts"][0]
    >>> s, e = out["output_spans"][0]
    >>> t[s:e]
    '4'
    """
    tmpl = template or ("### Instruction:\n{instruction}\n"
                        "### Input:\n{input}\n### Response:\n")
    texts = []
    spans = []
    for i, rec in enumerate(records):
        for key in ("instruction", "output"):
            if key not in rec:
                raise ValueError(f"record {i} is missing {key!r}.")
        head = tmpl.format(instruction=rec["instruction"],
                           input=rec.get("input", ""))
        out = str(rec["output"])
        texts.append(head + out)
        spans.append((len(head), len(head) + len(out)))
    if not texts:
        raise ValueError("no records supplied.")
    return RichResult(payload={
        "texts": texts, "output_spans": spans,
        "estimate": float(len(texts)), "n": len(texts),
        "method": "Instruction template with output loss mask (Alammar Ch 11)"})


def cheatsheet():
    return "allktmpl: rendered SFT text + the exact span the loss may see"
