# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 9.16: multimodal instruction-tuned prediction."""

from ._richresult import RichResult

__all__ = ["kamath_ch9_mm_instr_predict"]


def kamath_ch9_mm_instr_predict(I, M, theta, f=None):
    r"""A = f(I, M; theta).

    An instruction sample is the triplet (I, M, R): instruction,
    multimodal input, ground-truth response. Eq 9.16 is the forward
    pass, so the model is the caller's: pass a callable ``theta(I, M)``
    or supply ``f`` and let ``theta`` be its parameters, in which case
    ``f(I, M, theta)`` is called. The contract enforced is that a
    prediction actually comes back (not ``None``).

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, Eq 9.16, printed
    p. 391.

    Examples
    --------
    >>> out = kamath_ch9_mm_instr_predict(
    ...     "What animal?", "<image>", lambda i, m: "a cat")
    >>> out["answer"]
    'a cat'
    """
    if f is not None:
        if not callable(f):
            raise ValueError("f must be callable f(I, M, theta).")
        A = f(I, M, theta)
    elif callable(theta):
        A = theta(I, M)
    else:
        raise ValueError("give either a callable theta(I, M) or f= "
                         "with theta as its parameters.")
    if A is None:
        raise ValueError("the model returned no answer.")
    return RichResult(payload={
        "estimate": A, "answer": A, "instruction": I,
        "multimodal_input": M, "n": 1,
        "method": "multimodal instruction-tuned prediction "
                  "(Kamath Eq 9.16)"})


def cheatsheet():
    return "km144: A = f(I, M; theta) with the model contract enforced"
