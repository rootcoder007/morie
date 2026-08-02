# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 9.2: the input-alignment (text-generation) objective."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch9_input_alignment_loss"]


def kamath_ch9_input_alignment_loss(P_X, F_T, t, llm=None, loss_fn=None):
    r"""argmin over IN_ALIGN of L_txt-gen(LLM(P_X, F_T), t).

    ``llm(P_X, F_T)`` and ``loss_fn(prediction, t)`` are the caller's;
    both are required and both are contract-checked. ``P_X`` may be a
    single prompt-feature matrix (2-D), in which case the objective
    value is returned, or a stack of candidate projections (3-D), in
    which case the argmin over candidates is reported -- that is the
    "arg min" of Eq 9.2 made concrete over a finite candidate set.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, Eq 9.2, printed
    p. 380.

    Examples
    --------
    >>> import numpy as np
    >>> mse = lambda y, tgt: float(np.mean((np.asarray(y)
    ...                                     - np.asarray(tgt)) ** 2))
    >>> add = lambda p, f: np.asarray(p) + np.asarray(f)
    >>> out = kamath_ch9_input_alignment_loss(
    ...     [[[0.0]], [[1.0]]], [[1.0]], [[1.0]], llm=add, loss_fn=mse)
    >>> out["estimate"], out["argmin"]
    (0.0, 0)
    """
    if not callable(llm):
        raise ValueError("llm= must be a callable LLM(P_X, F_T).")
    if not callable(loss_fn):
        raise ValueError("loss_fn= must be a callable "
                         "L_txt-gen(prediction, t).")
    P = np.asarray(P_X, dtype=float)
    if P.ndim not in (2, 3):
        raise ValueError("P_X must be a 2-D prompt-feature matrix or a "
                         "3-D stack of candidates; got "
                         f"{P.ndim} dimensions.")
    cands = P if P.ndim == 3 else P[None, ...]
    losses = []
    for c in cands:
        L = float(loss_fn(llm(c, F_T), t))
        if not np.isfinite(L):
            raise ValueError("loss_fn returned a non-finite value.")
        losses.append(L)
    k = int(np.argmin(losses))
    return RichResult(payload={
        "estimate": float(losses[k]), "argmin": k, "losses": losses,
        "n_candidates": len(losses), "n": len(losses),
        "method": "input-alignment text-generation objective "
                  "(Kamath Eq 9.2)"})


def cheatsheet():
    return "km130: L_txt-gen(LLM(P_X, F_T), t), argmin over candidates"
