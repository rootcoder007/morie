# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""KV-cache compression for autoregressive LLM inference."""

from . import _array_core as np

from ._richresult import RichResult
from .hmint8 import geron_int8_quant

__all__ = ["geron_kv_cache_compress"]

_METHOD = "KV-cache quantization"


def geron_kv_cache_compress(K, V, n_bits=8, per_head=True, dtype_bytes=2):
    """
    KV-cache compression for autoregressive LLM inference.

    Formula: quantize stored K,V tensors; reduce memory during generation

    During generation the weights are read once per token but the KV
    cache grows linearly with the sequence, so past a few thousand
    tokens the cache -- not the model -- is what does not fit.
    Quantizing it is the cheapest fix available, and it is a pure
    inference-time change: nothing is retrained.

    The quantization is delegated to
    :func:`morie.fn.hmint8.geron_int8_quant`.  What this entry adds is
    the *granularity*, which is where the accuracy is won or lost:
    with ``per_head=True`` each head gets its own scale, so one head
    with large activations cannot flatten every other head's resolution.
    That per-head scale is a handful of extra floats against a tensor of
    thousands, and the memory accounting returned here includes them, so
    the reported ratio is honest rather than the nominal ``64/n_bits``.

    Parameters
    ----------
    K, V : array-like, shape (n_heads, seq_len, d_head) or (seq_len, d_head)
        Cached keys and values.
    n_bits : int
        Bit width.
    per_head : bool
        One scale per head, rather than one for the whole tensor.
    dtype_bytes : int
        Bytes per element before compression (2 = fp16).

    Returns
    -------
    result : RichResult
        Keys: K_dequantized, V_dequantized, bytes_before, bytes_after,
        compression_ratio, max_error, estimate, n, method.

    Examples
    --------
    Against an fp16 cache, 8 bits is a factor of two, not four -- the
    baseline is already 2 bytes -- and slightly under two once the
    per-head scales are counted:

    >>> rng = np.random.default_rng(0)
    >>> Kc = rng.normal(size=(4, 16, 8)); Vc = rng.normal(size=(4, 16, 8))
    >>> r = geron_kv_cache_compress(Kc, Vc, n_bits=8)
    >>> r["bytes_before"], r["bytes_after"]
    (2048, 1056)
    >>> bool(1.9 < r["compression_ratio"] < 2.0)
    True

    Against fp32 the same quantization is a factor of four:

    >>> f32 = geron_kv_cache_compress(Kc, Vc, n_bits=8, dtype_bytes=4)
    >>> bool(3.8 < f32["compression_ratio"] < 4.0)
    True

    The round trip is lossy but bounded, and the error is reported
    rather than assumed away:

    >>> bool(r["max_error"] < 0.05)
    True
    >>> r["K_dequantized"].shape
    (4, 16, 8)

    Per-head scaling beats a single global scale when one head is much
    louder than the others:

    >>> loud = np.ones((2, 8, 4)); loud[0] *= 1000.0
    >>> a = geron_kv_cache_compress(loud, loud, n_bits=8, per_head=True)
    >>> b = geron_kv_cache_compress(loud, loud, n_bits=8, per_head=False)
    >>> bool(a["max_error"] < b["max_error"])
    True

    References
    ----------
    Géron Ch 17
    """
    Ka = np.asarray(K, dtype=float)
    Va = np.asarray(V, dtype=float)
    if Ka.ndim == 2:
        Ka = Ka[None, :, :]
    if Va.ndim == 2:
        Va = Va[None, :, :]
    if Ka.ndim != 3 or Va.ndim != 3:
        raise ValueError(
            f"geron_kv_cache_compress: K and V must be (n_heads, seq_len, d_head) or (seq_len, d_head), "
            f"got ndim {Ka.ndim} and {Va.ndim}"
        )
    if Ka.shape != Va.shape:
        raise ValueError(f"geron_kv_cache_compress: K has shape {Ka.shape} but V has {Va.shape}")
    if Ka.size == 0:
        raise ValueError("geron_kv_cache_compress: the cache is empty")
    if not np.all(np.isfinite(Ka)) or not np.all(np.isfinite(Va)):
        raise ValueError("geron_kv_cache_compress: K and V must be finite")
    db = int(dtype_bytes)
    if db < 1:
        raise ValueError(f"geron_kv_cache_compress: dtype_bytes must be at least 1, got {dtype_bytes!r}")
    b = int(n_bits)

    n_heads = Ka.shape[0]
    outs = {}
    scales = {"K": [], "V": []}
    max_err = 0.0
    for name, T in (("K", Ka), ("V", Va)):
        deq = np.empty_like(T)
        if per_head:
            for h in range(n_heads):
                if np.all(T[h] == 0):
                    deq[h] = T[h]
                    scales[name].append(0.0)
                    continue
                q = geron_int8_quant(T[h].ravel(), n_bits=b, symmetric=True)
                deq[h] = np.asarray(q["dequantized"]).reshape(T[h].shape)
                scales[name].append(float(q["scale"]))
        else:
            q = geron_int8_quant(T.ravel(), n_bits=b, symmetric=True)
            deq = np.asarray(q["dequantized"]).reshape(T.shape)
            scales[name].append(float(q["scale"]))
        outs[name] = deq
        max_err = max(max_err, float(np.max(np.abs(deq - T))))

    n_elem = int(Ka.size + Va.size)
    bytes_before = n_elem * db
    n_scales = len(scales["K"]) + len(scales["V"])
    bytes_after = int(np.ceil(n_elem * b / 8.0)) + n_scales * 4
    ratio = bytes_before / bytes_after

    return RichResult(
        title="KV-cache compression",
        summary_lines=[
            ("Heads x seq x d_head", " x ".join(str(s) for s in Ka.shape)),
            ("Bits", b),
            ("Bytes before / after", f"{bytes_before} / {bytes_after}"),
            ("Compression ratio", ratio),
            ("Max round-trip error", max_err),
        ],
        interpretation=(
            "The cache, not the weights, is what grows with the sequence; per-head scales cost a "
            "few floats and stop one loud head from flattening the rest."
        ),
        payload={
            "K_dequantized": outs["K"],
            "V_dequantized": outs["V"],
            "scales": scales,
            "bytes_before": bytes_before,
            "bytes_after": bytes_after,
            "compression_ratio": ratio,
            "max_error": max_err,
            "n_bits": b,
            "estimate": ratio,
            "n": n_elem,
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmkvc: KV-cache quantization (delegates to hmint8) with per-head scales and honest byte accounting"
