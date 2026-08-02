# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Top-p (nucleus) sampling: truncate to the smallest set with
cumulative probability >= p."""

from . import _array_core as np

from ._richresult import RichResult
from .toppd import top_p_nucleus

__all__ = ["kamath_nucleus_sampling"]


def kamath_nucleus_sampling(logits, p, T=1.0):
    """V_p = smallest {v_1, ..., v_k} with sum P(v_i) >= p; renormalise.

    Kamath's top-p is Holtzman's top-p, so the truncation is
    DELEGATED to ``morie.fn.toppd`` rather than reimplemented; this
    module adds the kept-token report (which ids survived, and the
    mass they carried before renormalising) that a sampler needs to
    explain itself.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, top-p sampling.

    Examples
    --------
    >>> out = kamath_nucleus_sampling([0.0, 0.0, 5.0], 0.5)
    >>> out["n_kept"]
    1
    >>> out["kept"]
    [2]
    >>> out["probabilities"][2]
    1.0
    >>> flat = kamath_nucleus_sampling([0.0, 0.0], 1.0)
    >>> flat["probabilities"]
    [0.5, 0.5]
    """
    z = np.atleast_1d(np.asarray(logits, dtype=float)).ravel()
    if z.size == 0:
        raise ValueError("no logits supplied.")
    if not np.all(np.isfinite(z)):
        raise ValueError("logits must be finite.")
    if not 0.0 < float(p) <= 1.0:
        raise ValueError(f"p must lie in (0, 1]; got {p}.")
    if float(T) <= 0:
        raise ValueError(
            f"the temperature must be positive; got {T}. T -> 0 is "
            "greedy decoding, which is a different function.")
    base = top_p_nucleus(z, p=float(p), T=float(T))
    keep = np.asarray(base["keep_mask"], dtype=bool)
    probs = np.asarray(base["tensor"], dtype=float)
    # Pre-truncation mass carried by the kept set.
    zz = z / float(T)
    zz = zz - zz.max()
    raw = np.exp(zz)
    raw = raw / raw.sum()
    return RichResult(payload={
        "probabilities": [float(v) for v in probs],
        "kept": [int(i) for i in np.flatnonzero(keep)],
        "n_kept": int(base["n_kept"]),
        "kept_mass": float(raw[keep].sum()),
        "estimate": float(probs.max()),
        "p": float(p), "temperature": float(T), "n": int(z.size),
        "method": "Nucleus (top-p) truncation (delegates to toppd)"})


def cheatsheet():
    return "kmnuc: toppd's top-p truncation plus the kept-token report"
