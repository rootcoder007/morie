# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hugging Face Pipelines: high-level inference wrapper."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_hf_pipelines"]

_METHOD = "Task pipeline (preprocess -> model -> postprocess)"

_TASKS = ("text-classification", "sentiment-analysis", "feature-extraction", "text-generation")


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def geron_hf_pipelines(task, inputs, model, labels=None, top_k=1):
    """
    Hugging Face Pipelines: high-level inference wrapper.

    Formula: pipeline(task)(inputs) -> predictions

    A pipeline is not a model; it is the *postprocessing contract* that
    turns raw model output into the shape a task expects.  That contract
    is what is implemented here, natively, with the model supplied by the
    caller -- there is no network call and no transformers dependency.

    Per task, the model's obligation and the postprocessing are:

    ``text-classification`` / ``sentiment-analysis``
        model returns logits of shape ``(n_inputs, n_labels)``;
        softmax, then argmax with the score attached.
    ``feature-extraction``
        model returns one vector per input; the pipeline only checks the
        widths agree and returns the stacked matrix.
    ``text-generation``
        model returns a string (or sequence) per input; passed through.

    The obligation is enforced.  A classifier that returns one row when
    given three inputs, or a ragged feature matrix, raises here rather
    than producing a silently misaligned prediction, which is the
    failure mode a "high-level wrapper" is otherwise very good at hiding.

    Parameters
    ----------
    task : str
        One of the tasks above.
    inputs : sequence
        Inputs, one per prediction.
    model : callable
        ``model(inputs) -> raw output`` as described per task.
    labels : sequence of str, optional
        Label names for the classification tasks; defaults to
        ``LABEL_0 ...``.
    top_k : int
        Number of ranked labels to return per input.

    Returns
    -------
    result : RichResult
        Keys: predictions, scores, raw, task, estimate, n, method.

    Examples
    --------
    Sentiment analysis over two inputs; the softmax of ``[2, 0]`` gives
    ``0.880797`` for the first label:

    >>> logits = lambda xs: [[2.0, 0.0], [0.0, 3.0]]
    >>> r = geron_hf_pipelines("sentiment-analysis", ["good", "bad"], logits,
    ...                        labels=["POSITIVE", "NEGATIVE"])
    >>> [p["label"] for p in r["predictions"]]
    ['POSITIVE', 'NEGATIVE']
    >>> round(float(r["predictions"][0]["score"]), 6)
    0.880797

    Scores are probabilities, so they sum to one per input:

    >>> [round(float(s), 9) for s in np.sum(r["scores"], axis=1)]
    [1.0, 1.0]

    Feature extraction just validates and stacks:

    >>> f = geron_hf_pipelines("feature-extraction", ["a", "b"],
    ...                        lambda xs: [[1.0, 2.0], [3.0, 4.0]])
    >>> f["predictions"].shape
    (2, 2)

    A model that returns the wrong number of rows is caught:

    >>> geron_hf_pipelines("text-classification", ["a", "b"], lambda xs: [[1.0, 0.0]])
    Traceback (most recent call last):
        ...
    ValueError: geron_hf_pipelines: model returned 1 rows for 2 inputs

    References
    ----------
    Géron Ch 14
    """
    if task not in _TASKS:
        raise ValueError(f"geron_hf_pipelines: task must be one of {list(_TASKS)}, got {task!r}")
    if not callable(model):
        raise ValueError(f"geron_hf_pipelines: model must be callable, got {type(model).__name__}")
    try:
        items = list(inputs)
    except TypeError:
        raise ValueError("geron_hf_pipelines: inputs must be a sequence, one entry per prediction") from None
    if not items:
        raise ValueError("geron_hf_pipelines: inputs is empty")

    raw = model(items)

    if task in ("text-classification", "sentiment-analysis"):
        logits = np.atleast_2d(np.asarray(raw, dtype=float))
        if logits.shape[0] != len(items):
            raise ValueError(f"geron_hf_pipelines: model returned {logits.shape[0]} rows for {len(items)} inputs")
        if not np.all(np.isfinite(logits)):
            raise ValueError("geron_hf_pipelines: model returned non-finite logits")
        n_lab = logits.shape[1]
        names = [f"LABEL_{i}" for i in range(n_lab)] if labels is None else list(labels)
        if len(names) != n_lab:
            raise ValueError(f"geron_hf_pipelines: {len(names)} labels supplied but the model returned {n_lab} logits")
        k = int(top_k)
        if not (1 <= k <= n_lab):
            raise ValueError(f"geron_hf_pipelines: top_k must lie in 1..{n_lab}, got {top_k!r}")
        probs = _softmax(logits)
        preds = []
        for i in range(len(items)):
            order = np.argsort(probs[i])[::-1][:k]
            if k == 1:
                j = int(order[0])
                preds.append({"label": names[j], "score": float(probs[i, j])})
            else:
                preds.append([{"label": names[int(j)], "score": float(probs[i, int(j)])} for j in order])
        out = preds
        scores = probs
        headline = float(np.mean(np.max(probs, axis=1)))
    elif task == "feature-extraction":
        vectors = [np.atleast_1d(np.asarray(v, dtype=float)).ravel() for v in raw]
        if len(vectors) != len(items):
            raise ValueError(f"geron_hf_pipelines: model returned {len(vectors)} vectors for {len(items)} inputs")
        widths = {v.size for v in vectors}
        if len(widths) != 1:
            raise ValueError(f"geron_hf_pipelines: feature vectors have inconsistent widths {sorted(widths)}")
        out = np.vstack(vectors)
        if not np.all(np.isfinite(out)):
            raise ValueError("geron_hf_pipelines: model returned non-finite features")
        scores = None
        headline = float(np.mean(np.linalg.norm(out, axis=1)))
    else:  # text-generation
        gens = list(raw)
        if len(gens) != len(items):
            raise ValueError(f"geron_hf_pipelines: model returned {len(gens)} generations for {len(items)} inputs")
        out = [{"generated_text": g} for g in gens]
        scores = None
        headline = float(np.mean([len(str(g)) for g in gens]))

    return RichResult(
        title=f"Pipeline: {task}",
        summary_lines=[("Task", task), ("Inputs", len(items))],
        interpretation=(
            "The pipeline owns the postprocessing contract; enforcing it is what keeps a "
            "misaligned model output from becoming a silently wrong prediction."
        ),
        payload={
            "predictions": out,
            "scores": scores,
            "raw": raw,
            "task": task,
            "estimate": headline,
            "n": len(items),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmhfpi: task pipeline -- enforced model contract plus per-task postprocessing (softmax/stack/passthrough)"
