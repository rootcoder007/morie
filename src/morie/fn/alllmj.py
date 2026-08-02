# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""LLM-as-judge aggregation (Zheng et al. 2023; Alammar Ch 12)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["alammar_llm_as_judge"]


def alammar_llm_as_judge(responses, rubric, judge_model, n_samples=1):
    """score(y) = mean over judge samples of judge(rubric, y).

    The judge is a callable (rubric, response, sample_index) -> score.
    Per-response score VARIANCE across judge samples is reported: a
    judge that disagrees with itself is measuring noise, and hiding
    that behind a mean is how bad evaluations get published.

    References: Alammar and Grootendorst, Ch 12; Zheng et al. (2023).
    """
    if not callable(judge_model):
        raise ValueError("judge_model must be callable "
                         "(rubric, response, sample_index) -> score.")
    R = [str(r) for r in responses]
    if not R:
        raise ValueError("no responses supplied.")
    k = int(n_samples)
    if k < 1:
        raise ValueError("n_samples must be positive.")
    scores = np.array([[float(judge_model(rubric, r, s))
                        for s in range(k)] for r in R])
    means = scores.mean(axis=1)
    sds = scores.std(axis=1, ddof=1) if k > 1 else np.zeros(len(R))
    return RichResult(payload={
        "scores": [float(v) for v in means],
        "judge_sd": [float(v) for v in sds],
        "best_response": int(np.argmax(means)),
        "estimate": float(means.max()), "n": len(R),
        "method": "LLM-as-judge with self-disagreement reported "
                  "(Zheng et al. 2023)"})


def cheatsheet():
    return "alllmj: mean judge score per response, judge variance surfaced"
