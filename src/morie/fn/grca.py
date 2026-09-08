# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-attention: Q from the decoder, K and V from the encoder output."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_cross_attention"]

_METHOD = "Scaled dot-product cross-attention"


def _softmax_rows(Z):
    Z = Z - Z.max(axis=-1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=-1, keepdims=True)


def geron_cross_attention(X_dec, X_enc, WQ, WK, WV, mask=None):
    r"""Attend from decoder positions to encoder positions.

    .. math::
        \mathrm{CA}(X_{\text{dec}}, X_{\text{enc}}) =
        \operatorname{softmax}\!\left(
        \frac{X_{\text{dec}}W_Q\,(X_{\text{enc}}W_K)^{\top}}{\sqrt{d_k}}
        \right) X_{\text{enc}}W_V

    Queries come from the decoder, keys and values from the encoder --
    which is the entire difference from self-attention, and the reason
    cross-attention needs no causal mask: the encoder side is fully
    observed already.  The :math:`\sqrt{d_k}` divisor keeps the logits
    from growing with width and saturating the softmax.

    Parameters
    ----------
    X_dec : array-like, shape (Td, d_model)
    X_enc : array-like, shape (Te, d_model)
    WQ : array-like, shape (d_model, d_k)
    WK : array-like, shape (d_model, d_k)
    WV : array-like, shape (d_model, d_v)
    mask : array-like of bool, shape (Td, Te), optional
        ``True`` marks a *disallowed* pair (e.g. encoder padding); such
        logits are set to ``-inf`` before the softmax.

    Returns
    -------
    RichResult
        Payload keys ``output``, ``attention_weights``, ``logits``,
        ``scale``, ``d_k``, ``estimate`` (mean of the output), ``n``,
        ``method``.

    References
    ----------
    Géron Ch 15, Cross-Attention section.

    Examples
    --------
    A zero query makes every logit zero, so attention is uniform and the
    output is the mean value vector:

    >>> r = geron_cross_attention([[0.0]], [[1.0], [3.0]], WQ=[[0.0]],
    ...                           WK=[[1.0]], WV=[[1.0]])
    >>> [round(w, 6) for w in r["attention_weights"][0]]
    [0.5, 0.5]
    >>> [round(v, 6) for v in r["output"][0]]
    [2.0]

    Masking the second encoder position sends all the weight to the first:

    >>> r2 = geron_cross_attention([[0.0]], [[1.0], [3.0]], WQ=[[0.0]],
    ...                            WK=[[1.0]], WV=[[1.0]], mask=[[False, True]])
    >>> [round(w, 6) for w in r2["attention_weights"][0]]
    [1.0, 0.0]
    >>> [round(v, 6) for v in r2["output"][0]]
    [1.0]
    """
    Xd = np.atleast_2d(np.asarray(X_dec, dtype=float))
    Xe = np.atleast_2d(np.asarray(X_enc, dtype=float))
    WQ = np.atleast_2d(np.asarray(WQ, dtype=float))
    WK = np.atleast_2d(np.asarray(WK, dtype=float))
    WV = np.atleast_2d(np.asarray(WV, dtype=float))
    if Xd.size == 0 or Xe.size == 0:
        raise ValueError("X_dec and X_enc must be non-empty.")
    if Xd.shape[1] != WQ.shape[0]:
        raise ValueError(f"X_dec width {Xd.shape[1]} != WQ rows {WQ.shape[0]}.")
    if Xe.shape[1] != WK.shape[0]:
        raise ValueError(f"X_enc width {Xe.shape[1]} != WK rows {WK.shape[0]}.")
    if Xe.shape[1] != WV.shape[0]:
        raise ValueError(f"X_enc width {Xe.shape[1]} != WV rows {WV.shape[0]}.")
    if WQ.shape[1] != WK.shape[1]:
        raise ValueError(
            f"WQ and WK must map into the same d_k, got {WQ.shape[1]} and {WK.shape[1]}."
        )
    for name, arr in (("X_dec", Xd), ("X_enc", Xe), ("WQ", WQ), ("WK", WK), ("WV", WV)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values.")

    d_k = WQ.shape[1]
    scale = 1.0 / np.sqrt(d_k)
    Q = Xd @ WQ
    K = Xe @ WK
    V = Xe @ WV
    logits = (Q @ K.T) * scale
    if mask is not None:
        M = np.asarray(mask, dtype=bool)
        if M.shape != logits.shape:
            raise ValueError(f"mask must have shape {logits.shape}, got {M.shape}.")
        if np.all(M, axis=1).any():
            raise ValueError("mask blocks every encoder position for at least one query.")
        logits = np.where(M, -np.inf, logits)
    A = _softmax_rows(logits)
    out = A @ V

    return RichResult(
        title="Cross-attention",
        summary_lines=[("Decoder length", int(Xd.shape[0])),
                       ("Encoder length", int(Xe.shape[0])),
                       ("d_k", int(d_k))],
        payload={
            "output": out.tolist(),
            "attention_weights": A.tolist(),
            "logits": np.where(np.isfinite(logits), logits, -np.inf).tolist(),
            "scale": float(scale),
            "d_k": int(d_k),
            "estimate": float(out.mean()),
            "n": int(Xd.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grca: cross-attention softmax(Q_dec K_enc^T / sqrt(d_k)) V_enc"
