# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 3.3: answer search over the filled prompts."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch3_prompt_search_argmax"]


def _fill(x, z):
    if "[z]" in x:
        return x.replace("[z]", z)
    return f"{x}{z}"


def kamath_ch3_prompt_search_argmax(x, z, theta, f_fill=None):
    """z_hat = search_{z in Z} P(f_fill(x, z); theta).

    ``theta`` is the caller's scorer: a callable taking the FILLED
    prompt and returning a finite score (a probability, a log
    probability -- argmax is invariant to any monotone map). ``z`` is
    the candidate answer set Z. ``f_fill`` defaults to substituting the
    [z] slot, appending when the template has none.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 3, Eq 3.3, printed
    p. 94.

    Examples
    --------
    >>> score = lambda s: float(len(s))
    >>> out = kamath_ch3_prompt_search_argmax(
    ...     "Paris is [z].", ["great", "terrible"], score)
    >>> out["z_hat"], out["estimate"]
    ('terrible', 18.0)
    """
    if not isinstance(x, str) or not x:
        raise ValueError("x must be a non-empty prompt string.")
    cands = list(z)
    if not cands:
        raise ValueError("the candidate answer set Z is empty; there is "
                         "nothing to search over.")
    if not callable(theta):
        raise ValueError("theta must be a callable scorer "
                         "filled_prompt -> score.")
    fill = _fill if f_fill is None else f_fill
    filled, scores = [], []
    for c in cands:
        if not isinstance(c, str):
            raise ValueError("every candidate in z must be a string.")
        s = fill(x, c)
        v = float(theta(s))
        if not np.isfinite(v):
            raise ValueError(
                f"theta returned a non-finite score for {s!r}.")
        filled.append(s)
        scores.append(v)
    arr = np.asarray(scores, dtype=float)
    best = int(np.argmax(arr))
    return RichResult(payload={
        "estimate": float(arr[best]), "z_hat": cands[best],
        "filled_prompt": filled[best],
        "scores": {c: float(v) for c, v in zip(cands, arr)},
        "n": len(cands),
        "method": "argmax answer search over filled prompts "
                  "(Kamath Eq 3.3)"})


def cheatsheet():
    return "km044: z_hat = argmax_z score(f_fill(x, z))"
