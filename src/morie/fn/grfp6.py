# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""FP16 mixed-precision training with loss scaling."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_fp16_mixed_precision"]

_METHOD = "FP16 mixed precision with loss scaling"

_FP16_MAX = 65504.0
_FP16_TINY = 6.103515625e-05      # smallest normal
_FP16_SUBNORMAL_MIN = 5.960464477539063e-08


def geron_fp16_mixed_precision(loss, S, gradients=None):
    r"""Scale the loss up so small gradients survive FP16.

    .. math::
        \text{loss}_{\text{scaled}} = \text{loss} \times S,
        \qquad g_{\text{FP32}} = g_{\text{FP16}} / S

    FP16's problem is not the top of the range but the bottom: the
    smallest normal value is ``6.1e-5``, and a gradient below that
    flushes to zero and is simply lost.  Multiplying the loss by ``S``
    multiplies every gradient by ``S`` (the chain rule is linear),
    lifting them into representable range; dividing by ``S`` before the
    FP32 master-weight update takes the scaling straight back out, so
    the mathematics is unchanged.

    The failure mode at the other end is real too, which is why ``S`` is
    halved on overflow in practice: this function reports
    ``overflow`` when the scaled loss or any scaled gradient passes
    FP16's maximum of 65504, and ``n_underflow`` for gradients that
    would still flush to zero *after* scaling.

    Parameters
    ----------
    loss : float
        Unscaled loss, finite and non-negative.
    S : float
        Loss scale, positive (typically a power of two).
    gradients : array-like, optional
        Unscaled gradients; when given, the scaled and recovered
        versions are reported alongside the round-trip error.

    Returns
    -------
    RichResult
        Payload keys ``loss_scaled``, ``overflow``, ``scaled_gradients``,
        ``recovered_gradients``, ``max_roundtrip_error``,
        ``n_underflow_before``, ``n_underflow_after``, ``fp16_max``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 17, Mixed Precision / FP16 section (Micikevicius et al.
    2018).

    Examples
    --------
    Scaling by 1024 lifts the loss and does not overflow:

    >>> r = geron_fp16_mixed_precision(0.001, 1024.0)
    >>> r["loss_scaled"]
    1.024
    >>> r["overflow"]
    False

    A gradient of ``1e-8`` is below FP16's smallest normal and would be
    lost; after scaling it is representable, and dividing back recovers
    it exactly:

    >>> r2 = geron_fp16_mixed_precision(0.001, 1024.0, gradients=[1e-8, 0.5])
    >>> r2["n_underflow_before"], r2["n_underflow_after"]
    (1, 0)
    >>> r2["max_roundtrip_error"]
    0.0

    Too large a scale overflows FP16's 65504 ceiling, which is the
    signal to halve ``S``:

    >>> geron_fp16_mixed_precision(100.0, 65536.0)["overflow"]
    True
    """
    loss = float(loss)
    if not np.isfinite(loss):
        raise ValueError(f"loss must be finite, got {loss}.")
    if loss < 0:
        raise ValueError(f"loss must be non-negative, got {loss}.")
    S = float(S)
    if not np.isfinite(S) or S <= 0:
        raise ValueError(f"S must be a positive finite loss scale, got {S}.")

    ls = loss * S
    overflow = bool(ls > _FP16_MAX)

    sg = rg = None
    err = 0.0
    n_before = n_after = 0
    if gradients is not None:
        g = np.asarray(gradients, dtype=float)
        if not np.all(np.isfinite(g)):
            raise ValueError("gradients must be finite before scaling.")
        scaled = g * S
        recovered = scaled / S
        overflow = overflow or bool(np.any(np.abs(scaled) > _FP16_MAX))
        n_before = int(np.sum((g != 0) & (np.abs(g) < _FP16_TINY)))
        n_after = int(np.sum((scaled != 0) & (np.abs(scaled) < _FP16_SUBNORMAL_MIN)))
        err = float(np.max(np.abs(recovered - g))) if g.size else 0.0
        sg, rg = scaled.tolist(), recovered.tolist()

    return RichResult(
        title="FP16 mixed precision",
        summary_lines=[("Scaled loss", ls), ("S", S), ("Overflow", overflow)],
        payload={
            "loss_scaled": ls,
            "overflow": overflow,
            "scaled_gradients": sg,
            "recovered_gradients": rg,
            "max_roundtrip_error": err,
            "n_underflow_before": n_before,
            "n_underflow_after": n_after,
            "fp16_max": _FP16_MAX,
            "fp16_min_normal": _FP16_TINY,
            "S": S,
            "estimate": ls,
            "n": 1 if gradients is None else int(np.asarray(gradients).size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grfp6: loss*S lifts gradients above FP16's 6.1e-5 floor; divide by S before the FP32 update"
