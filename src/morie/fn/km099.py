# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.23: Expected Maximum Toxicity."""

from . import _array_core as np

from ._richresult import RichResult
from .km101 import _scores

__all__ = ["kamath_ch6_emt_metric"]


def kamath_ch6_emt_metric(Yhat, c):
    """EMT(Yhat) = max_{Yhat} c(Yhat).

    The WORST generation, not the average one: a model that is usually
    clean but occasionally vile scores badly here and well under Toxic
    Fraction, which is exactly why both are reported. ``c`` is the
    toxicity scorer (a callable or a sequence of scores in [0, 1]),
    validated by the shared checker km101 also uses.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.23, printed
    p. 250.

    Examples
    --------
    >>> out = kamath_ch6_emt_metric(["a", "b"],
    ...                             lambda y: {"a": 0.2, "b": 0.7}[y])
    >>> out["estimate"], out["argmax"]
    (0.7, 'b')
    """
    arr, outs = _scores(Yhat, c)
    k = int(np.argmax(arr))
    return RichResult(payload={
        "estimate": float(arr[k]), "argmax": outs[k], "argmax_index": k,
        "scores": [float(v) for v in arr], "n": len(outs),
        "method": "Expected Maximum Toxicity (Kamath Eq 6.23)"})


def cheatsheet():
    return "km099: max toxicity over the generations"
