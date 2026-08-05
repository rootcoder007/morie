# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Flamingo gated cross-attention.

DUPLICATE: this is the same method as :mod:`morie.fn.grflam`
(``geron_flamingo_cross_modal_attn``), which is already implemented
against Alayrac et al (2022).  Per ledger/wave2/DUPMAP.tsv this module
aliases that implementation instead of carrying a second copy of the
arithmetic.
"""

from ._richresult import RichResult
from .grflam import geron_flamingo_cross_modal_attn

__all__ = ["flamingo_gated_cross"]


def flamingo_gated_cross(x, vision, gate, weights=None, mask=None):
    """
    Flamingo gated cross-attention

    Formula: h <- h + tanh(g) * CrossAttn(h, vision).

    Thin alias of :func:`morie.fn.grflam.geron_flamingo_cross_modal_attn`
    -- see that module for the derivation.  When ``weights`` is omitted
    the three projections default to the identity, which makes the gated
    branch plain scaled dot-product cross-attention on the raw features.

    Parameters
    ----------
    x : array-like, shape (T, d_model)
        Language hidden states.
    vision : array-like, shape (Tv, d_model)
        Visual features.
    gate : float
        Gate parameter g; tanh(g) scales the branch, so g = 0 is exactly
        the identity on x.
    weights : sequence, optional
        (WQ, WK, WV) projections; identity by default.
    mask : array-like, optional
        Attention mask passed straight through.

    Returns
    -------
    result : dict
        Whatever ``geron_flamingo_cross_modal_attn`` returns, plus
        ``estimate`` (the mean of the updated hidden states) for the
        common result shape.

    References
    ----------
    Alayrac et al (2022), NeurIPS 35:23716-23736 (Flamingo).
    """
    rows = [list(r) if isinstance(r, (list, tuple)) else [r] for r in x]
    d = len(rows[0]) if rows else 0
    if d == 0:
        raise ValueError("empty input: x has no columns")
    if weights is None:
        eye = [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]
        weights = (eye, eye, eye)
    res = geron_flamingo_cross_modal_attn(x, vision, gate, weights, mask=mask)
    hn = res["h_new"]
    flat = [float(v) for r in hn for v in (r if isinstance(r, (list, tuple)) else [r])]
    out = dict(res)
    out["estimate"] = sum(flat) / len(flat)
    out["n"] = len(rows)
    out["method"] = "Flamingo gated cross-attention"
    return RichResult(payload=out)


def cheatsheet():
    return "flmgcr: Flamingo gated cross-attention (alias of grflam)"


# compact alias per ledger/NAMING.md
flamingogatedcross = flamingo_gated_cross
