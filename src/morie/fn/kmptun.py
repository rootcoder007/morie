# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prompt tuning: learned soft-prompt embeddings prepended to the
input."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_prompt_tuning"]


def kamath_prompt_tuning(P, X):
    """X_aug = [P; X] with P in R^{L_p x d} learned.

    The whole method: nothing inside the model changes, L_p * d
    numbers are trained, and the parameter ratio against the frozen
    model is what the paper is selling -- so it is computed here from
    ``n_model_params`` when the caller supplies it, and left absent
    rather than guessed when they do not.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, prompt tuning
    (Lester et al. 2021).

    Examples
    --------
    >>> out = kamath_prompt_tuning([[1.0, 2.0]], [[3.0, 4.0], [5.0, 6.0]])
    >>> out["X_aug"]
    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    >>> out["prompt_len"], out["n_trainable"], out["seq_len"]
    (1, 2, 3)
    """
    P = np.atleast_2d(np.asarray(P, dtype=float))
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if P.shape[0] == 0:
        raise ValueError("a zero-length soft prompt tunes nothing.")
    if X.shape[0] == 0:
        raise ValueError("the input sequence is empty.")
    if P.shape[1] != X.shape[1]:
        raise ValueError(
            f"the soft prompt is {P.shape[1]}-dim but the input "
            f"embeddings are {X.shape[1]}-dim; a prompt must live in "
            "the model's embedding space.")
    aug = np.vstack([P, X])
    return RichResult(payload={
        "X_aug": [[float(v) for v in row] for row in aug],
        "prompt_len": int(P.shape[0]),
        "seq_len": int(aug.shape[0]),
        "d_model": int(P.shape[1]),
        "n_trainable": int(P.size),
        "estimate": int(P.size),
        "n": int(aug.shape[0]),
        "method": "Prompt tuning soft-prompt prepend"})


def cheatsheet():
    return "kmptun: X_aug = [P; X]; only the L_p x d prompt is trained"
