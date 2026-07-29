# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Flamingo gated cross-attention."""

import numpy as np

from ._richresult import RichResult
from .grca import geron_cross_attention

__all__ = ["geron_flamingo_cross_modal_attn"]

_METHOD = "Flamingo tanh-gated cross-attention"


def geron_flamingo_cross_modal_attn(h, visual_features, alpha, weights, mask=None):
    r"""Inject visual information into language hidden states, gated.

    .. math::
        h \leftarrow h + \tanh(\alpha)\,
        \mathrm{CrossAttn}(h, \text{visual}),\qquad
        \alpha \text{ learned, one scalar per layer}

    The gate is initialised at :math:`\alpha = 0`, where
    :math:`\tanh 0 = 0` and the whole cross-attention branch
    contributes *nothing*.  That is the design: a frozen language model
    keeps behaving exactly as it did on day one, and the visual pathway
    fades in only as training moves ``alpha`` away from zero.  Since
    ``tanh`` is bounded by 1, the branch can never overwhelm the
    residual stream either.

    The attention itself is delegated to
    :func:`morie.fn.grca.geron_cross_attention` -- queries from the
    language tokens, keys and values from the visual features.

    Parameters
    ----------
    h : array-like, shape (T, d_model)
        Language hidden states.
    visual_features : array-like, shape (Tv, d_model)
    alpha : float
        Learned gate scalar (pre-tanh).
    weights : dict or sequence
        ``WQ``, ``WK``, ``WV`` -- as a dict with those keys or a
        3-sequence in that order. ``WV`` must map to ``d_model`` so the
        branch can be added to the residual stream.
    mask : array-like of bool, optional
        Passed through to the cross-attention.

    Returns
    -------
    RichResult
        Payload keys ``h_new``, ``gate`` (``tanh(alpha)``),
        ``attention_output``, ``attention_weights``, ``delta_norm``,
        ``is_identity`` (True when the gate is exactly 0),
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 16, Flamingo section (Alayrac et al. 2022).

    Examples
    --------
    At ``alpha = 0`` the layer is the identity, whatever the images say:

    >>> I = [[1.0, 0.0], [0.0, 1.0]]
    >>> vis = [[5.0, 5.0], [-5.0, 3.0]]
    >>> r = geron_flamingo_cross_modal_attn([[1.0, 0.0]], vis, 0.0,
    ...                                     {"WQ": I, "WK": I, "WV": I})
    >>> r["h_new"]
    [[1.0, 0.0]]
    >>> r["gate"], r["is_identity"]
    (0.0, True)

    Open the gate and the visual branch is added, scaled by ``tanh``:

    >>> r2 = geron_flamingo_cross_modal_attn([[0.0, 0.0]], vis, 1.0,
    ...                                      {"WQ": I, "WK": I, "WV": I})
    >>> round(r2["gate"], 10)
    0.761594156

    With a zero query the attention is uniform, so the branch is the
    mean visual vector times the gate: ``tanh(1) * 0`` and
    ``tanh(1) * 4``:

    >>> [round(v, 6) for v in r2["h_new"][0]]
    [0.0, 3.046377]
    """
    H = np.atleast_2d(np.asarray(h, dtype=float))
    Vf = np.atleast_2d(np.asarray(visual_features, dtype=float))
    if H.ndim != 2 or Vf.ndim != 2:
        raise ValueError(f"h and visual_features must be 2-D, got {H.shape} and {Vf.shape}.")
    alpha = float(alpha)
    if not np.isfinite(alpha):
        raise ValueError(f"alpha must be finite, got {alpha}.")

    if isinstance(weights, dict):
        missing = [k for k in ("WQ", "WK", "WV") if k not in weights]
        if missing:
            raise ValueError(f"weights is missing {missing}; need WQ, WK and WV.")
        WQ, WK, WV = weights["WQ"], weights["WK"], weights["WV"]
    else:
        seq = list(weights)
        if len(seq) != 3:
            raise ValueError(f"weights must hold exactly WQ, WK, WV; got {len(seq)} items.")
        WQ, WK, WV = seq
    if np.atleast_2d(np.asarray(WV, dtype=float)).shape[1] != H.shape[1]:
        raise ValueError(
            f"WV must map to d_model = {H.shape[1]} so the gated branch can be added "
            f"to the residual stream, got {np.atleast_2d(np.asarray(WV)).shape}."
        )

    ca = geron_cross_attention(H, Vf, WQ, WK, WV, mask=mask)
    attn = np.asarray(ca["output"], dtype=float)
    gate = float(np.tanh(alpha))
    H_new = H + gate * attn

    return RichResult(
        title="Flamingo gated cross-attention",
        summary_lines=[("alpha", alpha), ("gate = tanh(alpha)", gate),
                       ("||delta||", float(np.linalg.norm(gate * attn)))],
        payload={
            "h_new": H_new.tolist(),
            "gate": gate,
            "alpha": alpha,
            "attention_output": ca["output"],
            "attention_weights": ca["attention_weights"],
            "delta_norm": float(np.linalg.norm(gate * attn)),
            "is_identity": bool(gate == 0.0),
            "estimate": H_new.tolist(),
            "n": int(H.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grflam: h += tanh(alpha) * crossattn(h, visual) via grca; alpha=0 leaves the LM untouched"
