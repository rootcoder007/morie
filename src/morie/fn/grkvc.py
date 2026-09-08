# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""KV-cache size and compression."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_kv_cache_compression"]

_METHOD = "KV-cache memory footprint"


def geron_kv_cache_compression(seq_len, num_layers, num_heads, d_head, bits=16,
                               batch_size=1, baseline_bits=16):
    r"""How much memory the key/value cache eats.

    .. math::
        \text{cache\_bytes} = \text{seq\_len} \times \text{num\_layers}
        \times \text{num\_heads} \times d_{\text{head}} \times 2
        \times \frac{\text{bits}}{8}

    The factor 2 is keys *and* values.  Everything else is linear, and
    the linearity in ``seq_len`` is the whole problem: the cache grows
    with every token generated, so a long conversation ends up costing
    more memory than the weights it is running on.  That is why the
    lever people reach for is ``bits`` -- INT8 or INT4 quantization of
    the cache -- rather than architecture.

    ``compression_ratio`` compares against ``baseline_bits`` (FP16 by
    default), so it is exactly ``baseline_bits / bits``.

    Parameters
    ----------
    seq_len, num_layers, num_heads, d_head : int
        Positive.
    bits : int, optional
        Bits per stored value, default 16.
    batch_size : int, optional
        Sequences cached at once, default 1.
    baseline_bits : int, optional
        Precision to compare against, default 16.

    Returns
    -------
    RichResult
        Payload keys ``cache_bytes``, ``megabytes``, ``gigabytes``,
        ``baseline_bytes``, ``compression_ratio``, ``bytes_per_token``,
        ``n_values``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 17, KV-cache compression section.

    Examples
    --------
    A 1024-token context on a 32-layer, 32-head model with
    ``d_head = 128``, in FP16, is half a gigabyte:

    >>> r = geron_kv_cache_compression(1024, 32, 32, 128, bits=16)
    >>> r["cache_bytes"]
    536870912
    >>> r["megabytes"]
    512.0

    Quantizing the cache to INT4 divides it by four:

    >>> r2 = geron_kv_cache_compression(1024, 32, 32, 128, bits=4)
    >>> r2["megabytes"], r2["compression_ratio"]
    (128.0, 4.0)

    And the cost is linear in the context, per token:

    >>> r["bytes_per_token"]
    524288
    """
    vals = {"seq_len": seq_len, "num_layers": num_layers,
            "num_heads": num_heads, "d_head": d_head}
    ints = {}
    for name, v in vals.items():
        iv = int(v)
        if iv < 1:
            raise ValueError(f"{name} must be a positive integer, got {v}.")
        ints[name] = iv
    bits = int(bits)
    if bits < 1:
        raise ValueError(f"bits must be a positive integer, got {bits}.")
    baseline_bits = int(baseline_bits)
    if baseline_bits < 1:
        raise ValueError(f"baseline_bits must be a positive integer, got {baseline_bits}.")
    bs = int(batch_size)
    if bs < 1:
        raise ValueError(f"batch_size must be a positive integer, got {bs}.")

    n_values = (ints["seq_len"] * ints["num_layers"] * ints["num_heads"]
                * ints["d_head"] * 2 * bs)
    nbytes = n_values * bits // 8
    if n_values * bits % 8:
        nbytes += 1
    baseline = n_values * baseline_bits // 8

    return RichResult(
        title="KV-cache footprint",
        summary_lines=[("Bytes", int(nbytes)), ("MB", nbytes / 2**20),
                       ("bits", bits)],
        payload={
            "cache_bytes": int(nbytes),
            "megabytes": float(nbytes) / 2**20,
            "gigabytes": float(nbytes) / 2**30,
            "baseline_bytes": int(baseline),
            "compression_ratio": float(baseline_bits) / float(bits),
            "bytes_per_token": int(nbytes // ints["seq_len"]),
            "n_values": int(n_values),
            "bits": bits,
            "estimate": int(nbytes),
            "n": int(ints["seq_len"]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grkvc: bytes = seq*L*H*d*2*bits/8; linear in context, so quantize the bits"
