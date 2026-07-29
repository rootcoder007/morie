# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 9.14: the image-conditioned text generation (ITG) loss."""

from ._richresult import RichResult
from .km145 import kamath_ch9_mmllm_autoregressive

__all__ = ["kamath_ch9_itg_loss"]


def kamath_ch9_itg_loss(x, y):
    r"""L_ITG = -sum_{(x,y)} log prod_t P(y_t | y_<t, x).

    ``y`` is one sequence of per-token conditional probabilities per
    (visual context, text) pair; ``x`` is the matching list of visual
    contexts, used to check the pairing (pass ``None`` to skip). The
    inner -log of a token-probability product is exactly Eq 9.17, so
    each pair is scored by ``morie.fn.km145`` and the results summed.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, Eq 9.14, printed
    p. 389.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch9_itg_loss(None, [[0.5, 0.5], [0.25]])
    >>> abs(out["estimate"] - 2 * math.log(4)) < 1e-12
    True
    """
    seqs = list(y)
    if len(seqs) == 0:
        raise ValueError("no (x, y) pairs were given.")
    if x is not None:
        ctx = list(x)
        if len(ctx) != len(seqs):
            raise ValueError(
                f"{len(ctx)} visual contexts for {len(seqs)} text "
                "sequences; the pairs do not line up.")
    per_pair = [float(kamath_ch9_mmllm_autoregressive(s, None)["estimate"])
                for s in seqs]
    return RichResult(payload={
        "estimate": float(sum(per_pair)), "per_pair": per_pair,
        "n": len(per_pair),
        "method": "image-conditioned text generation loss "
                  "(Kamath Eq 9.14; the per-pair core in km145)"})


def cheatsheet():
    return "km142: summed per-pair response NLL over the (X, Y) corpus"
