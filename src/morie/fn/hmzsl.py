# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Zero-shot learning: LLM generalizes to unseen tasks from prompt only."""

from . import _array_core as np

from ._richresult import RichResult
from .hmsftm import geron_softmax_function

__all__ = ["geron_zero_shot"]


def geron_zero_shot(model, prompt, labels=None, null_prompt=None):
    """
    Zero-shot learning: LLM generalizes to unseen tasks from prompt only.

    Formula: P(y | prompt) without task-specific training

    Scores each candidate label under the model *as it is* -- no weights
    are updated, which is the whole definition of zero-shot. ``model``
    must return one score (a log-probability or logit) per label, either
    as ``model(prompt, labels)`` or as ``model(prompt)``. Scores are
    normalised with :func:`morie.fn.hmsftm.geron_softmax_function`.

    Optionally applies contextual calibration: with `null_prompt` given,
    the model's scores for a content-free prompt are subtracted before
    normalising, which removes the label-order and surface-form bias that
    makes raw zero-shot scores lopsided.

    Parameters
    ----------
    model : callable
        ``model(prompt, labels) -> scores`` (preferred) or
        ``model(prompt) -> scores``; may also return a ``{label: score}``
        mapping, in which case `labels` may be omitted.
    prompt : object
        Prompt passed to the model unchanged.
    labels : sequence, optional
        Candidate labels/verbalizers (>= 2) unless the model returns a mapping.
    null_prompt : object, optional
        Content-free prompt used for contextual calibration.

    Returns
    -------
    result : RichResult
        Keys: probabilities, predicted, predicted_label, margin, entropy,
        calibrated, estimate, n, method.

    Examples
    --------
    >>> scores = {"negative": 0.0, "positive": 1.0}
    >>> r = geron_zero_shot(lambda p: scores, "Review: it was great. Sentiment:")
    >>> r["predicted_label"]
    'positive'
    >>> [round(float(v), 6) for v in r["probabilities"]]
    [0.268941, 0.731059]
    >>> round(float(r["margin"]), 6)
    0.462117

    Calibration against a null prompt removes a constant label bias, so
    an equally-scored pair becomes exactly uniform:

    >>> f = lambda p: ([5.0, 0.0] if p == "" else [6.0, 1.0])
    >>> rc = geron_zero_shot(f, "x", labels=["a", "b"], null_prompt="")
    >>> [round(float(v), 6) for v in rc["probabilities"]]
    [0.5, 0.5]
    >>> bool(rc["calibrated"])
    True

    References
    ----------
    Géron Ch 15
    """
    if not callable(model):
        raise ValueError("geron_zero_shot: model must be a callable scoring the labels under the prompt")

    def _score(p):
        try:
            out = model(p, labels) if labels is not None else model(p)
        except TypeError:
            out = model(p)
        if isinstance(out, dict):
            keys = list(out.keys())
            return np.asarray([float(out[k]) for k in keys], dtype=float), keys
        return np.asarray(out, dtype=float).ravel(), None

    s, keys = _score(prompt)
    names = list(labels) if labels is not None else keys
    if names is None:
        raise ValueError(
            "geron_zero_shot: labels are required unless the model returns a {label: score} mapping"
        )
    if len(names) != s.size:
        raise ValueError(f"geron_zero_shot: model returned {s.size} scores but {len(names)} labels were given")
    if s.size < 2:
        raise ValueError(f"geron_zero_shot: zero-shot classification needs >= 2 candidate labels, got {s.size}")
    if not np.all(np.isfinite(s)):
        raise ValueError("geron_zero_shot: model returned non-finite scores")

    calibrated = False
    raw = s.copy()
    if null_prompt is not None:
        s0, _ = _score(null_prompt)
        if s0.size != s.size:
            raise ValueError(
                f"geron_zero_shot: null prompt produced {s0.size} scores but the prompt produced {s.size}"
            )
        if not np.all(np.isfinite(s0)):
            raise ValueError("geron_zero_shot: model returned non-finite scores for the null prompt")
        s = s - s0
        calibrated = True

    p = np.asarray(geron_softmax_function(s)["p"], dtype=float)
    order = np.argsort(-p)
    k = int(order[0])
    margin = float(p[order[0]] - p[order[1]])
    ent = float(-np.sum(p * np.log(np.maximum(p, np.finfo(float).tiny))))

    return RichResult(
        title="Zero-shot classification",
        summary_lines=[
            ("Labels", len(names)),
            ("Predicted", names[k]),
            ("Margin", margin),
            ("Entropy (nats)", ent),
        ],
        interpretation=(
            "No parameters change: the prompt does the work. Raw zero-shot scores carry a strong "
            "surface-form bias, so a null-prompt calibration is usually worth the extra call."
        ),
        payload={
            "probabilities": p,
            "scores": s,
            "raw_scores": raw,
            "labels": names,
            "predicted": k,
            "predicted_label": names[k],
            "margin": margin,
            "entropy": ent,
            "calibrated": calibrated,
            "estimate": float(p[k]),
            "n": int(len(names)),
            "method": "Zero-shot label scoring with softmax normalisation"
            + (" and null-prompt contextual calibration" if calibrated else ""),
        },
    )


def cheatsheet():
    return "hmzsl: Zero-shot learning: LLM generalizes to unseen tasks from prompt only"


# compact alias per ledger/NAMING.md
geronzeroshot = geron_zero_shot
