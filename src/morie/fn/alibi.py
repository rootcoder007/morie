# morie.fn -- wave 2 slice w2_00 (rootcoder007/morie)
"""ALiBi linear bias applied to precomputed attention scores.

This is the same method as :mod:`morie.fn.atalib` -- Press, Smith and
Lewis (2022), "Train short, test long", ICLR 2022, arXiv:2108.12409 --
entered at a different point: ``atalib`` takes Q, K and V and returns the
attention output, while this function takes the scores that have already
been formed and only adds the bias.

There is exactly one implementation.  The bias matrix and the slope
schedule are imported from ``atalib`` rather than written again here: a
second copy would agree with the first at 1e-9 forever and would be
indistinguishable from correct work while permanently doubling the
surface under a name that reads right.

The bias added is -m|i - j|; see ``atalib`` for why that coincides with
the paper's causal row on the lower triangle.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult

from .atalib import alibi_bias as _bias, head_slopes as _slopes

__all__ = ["alibi"]

head_slopes = _slopes
alibi_bias = _bias


def alibi(scores, slopes=None, causal=False):
    """Add the ALiBi penalty to an existing score matrix.

    Parameters
    ----------
    scores : array-like
        n_q by n_k pre-softmax scores, already scaled by 1/sqrt(d).
    slopes : float, optional
        The head slope m; defaults to the paper's single-head 2^-8.
    causal : bool
        Mask keys after the query position.

    Returns
    -------
    biased : the score matrix with -m|i - j| added
    bias : the penalty matrix itself
    """
    S = [[float(v) for v in r] for r in scores]
    nq = len(S)
    if nq == 0:
        raise ValueError("alibi: scores is empty")
    nk = len(S[0])
    if nk == 0:
        raise ValueError("alibi: scores is empty")
    for r in S:
        if len(r) != nk:
            raise ValueError("alibi: scores is ragged")
    m = 2.0 ** -8.0 if slopes is None else float(slopes)
    B = _bias(nq, nk, m, causal)
    out = [[S[i][j] + B[i][j] for j in range(nk)] for i in range(nq)]
    return RichResult(
        title="ALiBi linear bias",
        summary_lines=[("n_q", nq), ("n_k", nk), ("slope", m)],
        payload={
            "biased": out,
            "estimate": out[0][0],
            "bias": B,
            "slope": m,
            "n_q": nq,
            "n_k": nk,
            "causal": bool(causal),
            "method": "scores - m|i-j|; shared implementation with morie.fn.atalib",
        },
    )


def cheatsheet():
    return "alibi: ALiBi linear bias (shares atalib's implementation)"
