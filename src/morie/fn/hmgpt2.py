# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GPT-2: scaled-up decoder-only LM."""

import numpy as np

from ._richresult import RichResult
from .hmdctr import geron_decoder_only

__all__ = ["geron_gpt2"]

# The four released GPT-2 sizes (Radford et al. 2019): BPE vocabulary 50257,
# context window 1024 throughout; only depth and width change.
_SIZES = {
    "small": {"n_layers": 12, "n_heads": 12, "d_model": 768},
    "medium": {"n_layers": 24, "n_heads": 16, "d_model": 1024},
    "large": {"n_layers": 36, "n_heads": 20, "d_model": 1280},
    "xl": {"n_layers": 48, "n_heads": 25, "d_model": 1600},
}
_BASE = {"vocab_size": 50257, "max_len": 1024}


def geron_gpt2(X, n_layers=None, n_heads=None, size="small", **config):
    """
    GPT-2: scaled-up decoder-only LM.

    Formula: same as GPT-1 but larger (up to 1.5B params)

    The architecture is DELEGATED to
    :func:`morie.fn.hmdctr.geron_decoder_only`; this module supplies the
    four released configurations and, because the formula line is about
    *scale*, computes the scaling relationships rather than restating
    them:

    * ``params_vs_small`` -- how many times bigger this variant is;
    * ``non_embedding_params`` -- the number that actually scales with
      depth and width, since the embedding table is fixed by the 50257
      BPE vocabulary and dominates the small model;
    * ``all_sizes`` -- every variant's count, so the quadratic-in-width,
      linear-in-depth growth is visible directly.

    GPT-2 keeps GPT-1's objective and widens the context to 1024 tokens.
    Its width is not always divisible by common head counts -- ``xl`` uses
    25 heads of 64 -- which is why the divisibility check in ``hmdctr``
    matters when overriding ``n_heads``.

    Parameters
    ----------
    X : array-like
        Token ids.
    n_layers, n_heads : int, optional
        Override the variant's values.
    size : {"small", "medium", "large", "xl"}, default "small"
    **config
        Any other ``geron_decoder_only`` argument.

    Returns
    -------
    result : RichResult
        Keys: total_params, non_embedding_params, config, size,
        params_vs_small, all_sizes, d_head, estimate, n, method.

    Examples
    --------
    The four sizes, counted exactly with tied embeddings (the widely
    quoted 117M / 345M / 762M / 1.5B):

    >>> r = geron_gpt2([1, 2, 3])
    >>> r["total_params"]
    124439808
    >>> [geron_gpt2([1], size=s)["total_params"] for s in ("medium", "large", "xl")]
    [354823168, 774030080, 1557611200]

    Every head is 64 wide in every variant -- that is the constant the
    scaling holds fixed:

    >>> [geron_gpt2([1], size=s)["d_head"] for s in ("small", "medium", "large", "xl")]
    [64, 64, 64, 64]

    Non-embedding parameters scale far faster than the total, because the
    50257-row embedding table is a fixed cost:

    >>> r["non_embedding_params"]
    85056000
    >>> round(geron_gpt2([1], size="xl")["params_vs_small"], 3)
    12.517

    References
    ----------
    Géron Ch 15
    """
    if size not in _SIZES:
        raise ValueError(f"geron_gpt2: size must be one of {sorted(_SIZES)}, got {size!r}")
    cfg = dict(_BASE)
    cfg.update(_SIZES[size])
    cfg.update({k: v for k, v in config.items() if v is not None})
    if n_layers is not None:
        cfg["n_layers"] = int(n_layers)
    if n_heads is not None:
        cfg["n_heads"] = int(n_heads)

    arch = geron_decoder_only(X, **cfg)
    total = int(arch["total_params"])
    non_emb = total - int(arch["embedding_params"])

    all_sizes = {}
    for name, s in _SIZES.items():
        c = dict(_BASE)
        c.update(s)
        all_sizes[name] = int(geron_decoder_only([1], **c)["total_params"])

    return RichResult(
        title=f"GPT-2 ({size})",
        summary_lines=[("Parameters", total), ("Layers", cfg["n_layers"]), ("d_model", cfg["d_model"])],
        tables=[{"title": "Released sizes", "headers": ["variant", "params"], "rows": [[k, v] for k, v in all_sizes.items()]}],
        interpretation="Depth is linear in the parameter count, width quadratic; the embedding table is a fixed 50257-row cost.",
        payload={
            "total_params": total,
            "non_embedding_params": int(non_emb),
            "embedding_params": int(arch["embedding_params"]),
            "block_params": int(arch["block_params"]),
            "config": cfg,
            "size": size,
            "d_head": int(arch["d_head"]),
            "params_vs_small": float(total / all_sizes["small"]),
            "all_sizes": all_sizes,
            "mask": arch["mask"],
            "estimate": float(total),
            "n": int(cfg["n_layers"]),
            "method": "GPT-2 released configurations resolved through hmdctr, with scaling comparisons",
        },
    )


def cheatsheet():
    return "hmgpt2: GPT-2: scaled-up decoder-only LM"
