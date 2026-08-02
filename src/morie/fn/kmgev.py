# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""G-Eval: LLM-graded evaluation with probability-weighted score
aggregation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_g_eval"]


def _softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _rubric_scores(rubric):
    if isinstance(rubric, dict):
        scores = rubric.get("scores")
        if scores is None:
            raise ValueError(
                "a dict rubric must carry a 'scores' key listing the "
                "score points the judge may emit.")
    else:
        scores = rubric
    s = np.atleast_1d(np.asarray(scores, dtype=float)).ravel()
    if s.size < 2:
        raise ValueError("a rubric needs at least two score points.")
    if len(set(float(v) for v in s)) != s.size:
        raise ValueError("the rubric repeats a score point.")
    return s


def kamath_g_eval(x, y, rubric, model):
    """score = sum_s s * p(s), with p = softmax over the judge's logits
    for the digit tokens of the rubric's score points.

    ``model`` is the caller's judge: ``model(x, y, rubric)`` must
    return one logit per rubric score point, in the rubric's order.
    Nothing here calls an API -- the contract is enforced instead, so
    a judge returning the wrong shape fails loudly rather than being
    averaged away.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 8, G-Eval.

    Examples
    --------
    >>> out = kamath_g_eval("q", "a", [1, 2, 3], lambda x, y, r: [0.0] * 3)
    >>> out["estimate"]
    2.0
    >>> out = kamath_g_eval("q", "a", [1, 5],
    ...     lambda x, y, r: [0.0, float(np.log(3))])
    >>> abs(out["estimate"] - (1 * 0.25 + 5 * 0.75)) < 1e-12
    True
    """
    s = _rubric_scores(rubric)
    if not callable(model):
        raise ValueError(
            "model must be callable (x, y, rubric) -> logits over the "
            "rubric's score points.")
    logits = model(x, y, rubric)
    logits = np.atleast_1d(np.asarray(logits, dtype=float)).ravel()
    if logits.size != s.size:
        raise ValueError(
            f"the judge returned {logits.size} logits for {s.size} "
            "rubric score points.")
    if not np.all(np.isfinite(logits)):
        raise ValueError("the judge returned a non-finite logit.")
    p = _softmax(logits)
    score = float(np.dot(s, p))
    return RichResult(payload={
        "estimate": score,
        "probabilities": [float(v) for v in p],
        "score_points": [float(v) for v in s],
        "logits": [float(v) for v in logits],
        "n": int(s.size),
        "method": "G-Eval probability-weighted rubric score"})


def cheatsheet():
    return "kmgev: sum_s s * softmax(judge logits)_s over the rubric"
