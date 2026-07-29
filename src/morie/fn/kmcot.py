# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 4: zero-shot chain-of-thought prompting."""

from ._richresult import RichResult

__all__ = ["kamath_chain_of_thought"]


def kamath_chain_of_thought(prompt, model,
                            trigger="Let's think step by step.",
                            answer_marker="Answer:", parser=None):
    r"""y = parse(LLM(prompt + "Let us think step by step.")).

    ``model(prompt)`` is the caller's LLM. The reasoning is split from
    the answer either by ``parser(text) -> (reasoning, answer)`` or,
    by default, at ``answer_marker``. A generation with no marker is
    an ERROR, not an answer: returning the whole chain of thought as
    if it were the answer is how CoT pipelines silently corrupt
    downstream scoring.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, Chain-of-Thought;
    Kojima et al. (2022); Wei et al. (2022).

    Examples
    --------
    >>> out = kamath_chain_of_thought(
    ...     "2+2?", lambda p: "add 2 and 2. Answer: 4")
    >>> out["answer"], out["reasoning"]
    ('4', 'add 2 and 2.')
    >>> out["prompt"].endswith("Let's think step by step.")
    True
    """
    if not callable(model):
        raise ValueError("model must be callable model(prompt) -> text.")
    full = f"{prompt} {trigger}" if trigger else str(prompt)
    text = model(full)
    if text is None:
        raise ValueError("the model returned no text.")
    text = str(text)
    if parser is not None:
        if not callable(parser):
            raise ValueError("parser must be callable or None.")
        got = parser(text)
        if not (isinstance(got, tuple) and len(got) == 2):
            raise ValueError("parser must return (reasoning, answer).")
        reasoning, answer = got
    else:
        if answer_marker not in text:
            raise ValueError(
                f"the generation contains no {answer_marker!r}; the "
                "answer cannot be separated from the reasoning, and "
                "the whole chain of thought is not an answer.")
        head, _, tail = text.partition(answer_marker)
        reasoning, answer = head.strip(), tail.strip()
    return RichResult(payload={
        "estimate": answer, "answer": answer, "reasoning": reasoning,
        "prompt": full, "generation": text, "n": 1,
        "method": "zero-shot chain-of-thought prompting (Kamath Ch 4)"})


def cheatsheet():
    return "kmcot: append the trigger, then split reasoning from answer"
