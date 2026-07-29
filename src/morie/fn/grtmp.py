# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Temperature scaling of softmax logits before sampling."""

import numpy as np

from ._richresult import RichResult
from .grn021 import softmax_vector

__all__ = ["geron_temperature_sampling"]

_METHOD = "Temperature-scaled softmax"


def geron_temperature_sampling(logits, T=1.0):
    r"""Divide the logits by ``T``, then normalise.

    .. math::
        p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}

    One knob between greedy and uniform.  ``T -> 0`` puts all mass on the
    argmax; ``T = 1`` is the model's own distribution; ``T -> inf``
    flattens to uniform.  Temperature is monotone in entropy, and the
    ranking never changes -- scaling logits cannot reorder them -- so
    temperature buys diversity, not different preferences.  ``T = 0`` is
    rejected rather than silently treated as greedy: the formula divides
    by it.  Softmax itself is delegated to :mod:`morie.fn.grn021`.

    Parameters
    ----------
    logits : array-like, shape (V,)
    T : float
        Temperature, strictly positive.

    Returns
    -------
    RichResult
        Payload keys ``probabilities``, ``entropy`` (nats), ``argmax``,
        ``perplexity``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 15, Temperature sampling.

    Examples
    --------
    Logits ``[2, 1, 0]`` at ``T = 1``:

    >>> r = geron_temperature_sampling([2.0, 1.0, 0.0])
    >>> [round(p, 6) for p in r["probabilities"]]
    [0.665241, 0.244728, 0.090031]

    Halving the temperature sharpens; doubling it flattens, and entropy
    orders the three:

    >>> lo = geron_temperature_sampling([2.0, 1.0, 0.0], T=0.5)["entropy"]
    >>> hi = geron_temperature_sampling([2.0, 1.0, 0.0], T=2.0)["entropy"]
    >>> lo < r["entropy"] < hi
    True

    >>> geron_temperature_sampling([2.0, 1.0, 0.0], T=0.0)
    Traceback (most recent call last):
        ...
    ValueError: T must be strictly positive (the formula divides by it), got 0.0.
    """
    z = np.asarray(logits, dtype=float).ravel()
    if z.size == 0:
        raise ValueError("logits is empty.")
    if not np.all(np.isfinite(z)):
        raise ValueError("logits contains non-finite values.")
    T = float(T)
    if not np.isfinite(T) or T <= 0:
        raise ValueError(f"T must be strictly positive (the formula divides by it), got {T}.")

    p = softmax_vector(z / T)
    ent = float(-np.sum(p * np.log(np.maximum(p, 1e-300))))

    return RichResult(
        title="Temperature sampling",
        summary_lines=[("T", T), ("Entropy (nats)", ent)],
        payload={
            "probabilities": p.tolist(),
            "entropy": ent,
            "perplexity": float(np.exp(ent)),
            "argmax": int(np.argmax(p)),
            "temperature": T,
            "estimate": p.tolist(),
            "n": int(z.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grtmp: p = softmax(z/T); T->0 greedy, T->inf uniform, ranking never changes; T=0 raises"
