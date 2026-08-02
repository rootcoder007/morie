# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pyramid Vision Transformer: hierarchical multi-scale stages."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_pvt"]


def _lcg_matrix(rows, cols, seed, scale):
    s = int(seed) % 2**32
    out = np.empty(rows * cols)
    for i in range(rows * cols):
        s = (1664525 * s + 1013904223) % 2**32
        out[i] = ((s + 0.5) / 2**32 * 2.0 - 1.0) * scale
    return out.reshape(rows, cols)


def geron_pvt(image, stage_cfgs, seed=0):
    """
    Pyramid Vision Transformer (PVT): a multi-scale transformer.

    Formula: hierarchical transformer stages with shrinking spatial resolution

    ViT holds one resolution from first layer to last, which makes dense
    prediction impossible and attention quadratic in the token count.
    PVT reintroduces the CNN's pyramid: each stage re-patchifies its
    predecessor's output, so the grid shrinks by the patch size and the
    channel width grows, exactly as a ResNet's stages do.

    Spatial-reduction attention is the second half. Keys and values are
    pooled by ``sr_ratio`` before attention, so a stage costs
    N * N / r^2 instead of N^2 -- which is what makes the large early
    grids affordable at all. Both numbers are reported per stage.

    The patch embedding is executed on the concrete ``image``, so the
    per-stage grids, token counts and parameter counts are resolved
    against real data rather than asserted.

    Parameters
    ----------
    image : array-like, shape (H, W, C)
        Input image (a 2-D array is treated as one channel).
    stage_cfgs : sequence of mappings
        One per stage: ``patch_size``, ``dim``, optional ``heads``
        (default 1), ``sr_ratio`` (default 1) and ``W`` (the projection
        matrix; a reproducible LCG matrix is used when absent).
    seed : int, default 0
        LCG seed for the default projections.

    Returns
    -------
    result : RichResult
        Keys: tokens, stages, output_shape, n_parameters,
        attention_cost, full_attention_cost, estimate, n, method.

    Examples
    --------
    A 4x4x3 image with 2x2 patches and a projection that averages the
    patch: the first token is the mean of the 12 values in the top-left
    patch, (0+..+5 + 12+..+17)/12 = 8.5.

    >>> img = np.arange(48.0).reshape(4, 4, 3)
    >>> W = np.ones((12, 1)) / 12.0
    >>> r = geron_pvt(img, [{"patch_size": 2, "dim": 1, "W": W}])
    >>> r["output_shape"], float(r["tokens"][0, 0, 0])
    ((2, 2, 1), 8.5)

    Two stages halve the grid twice and widen the channels:

    >>> r2 = geron_pvt(np.arange(8 * 8 * 3.0).reshape(8, 8, 3),
    ...                [{"patch_size": 2, "dim": 4}, {"patch_size": 2, "dim": 8}])
    >>> [s["grid"] for s in r2["stages"]], [int(s["tokens"]) for s in r2["stages"]]
    ([(4, 4), (2, 2)], [16, 4])

    Spatial reduction cuts the attention cost by r^2:

    >>> r3 = geron_pvt(np.zeros((8, 8, 1)), [{"patch_size": 2, "dim": 4, "sr_ratio": 2}])
    >>> int(r3["attention_cost"]), int(r3["full_attention_cost"])
    (64, 256)

    References
    ----------
    Geron Ch 16
    """
    img = np.asarray(image, dtype=float)
    if img.ndim == 2:
        img = img[:, :, None]
    if img.ndim != 3 or img.size == 0:
        raise ValueError(f"geron_pvt: image must be (H, W, C) or (H, W), got shape {img.shape}")
    if not np.all(np.isfinite(img)):
        raise ValueError("geron_pvt: image contains non-finite values")
    cfgs = list(stage_cfgs)
    if not cfgs:
        raise ValueError("geron_pvt: stage_cfgs is empty")

    x = img
    stages = []
    params = 0
    cost = 0
    full = 0
    for i, cfg in enumerate(cfgs):
        if not hasattr(cfg, "get"):
            raise ValueError(f"geron_pvt: stage {i} must be a mapping, got {type(cfg).__name__}")
        p = int(cfg.get("patch_size", 2))
        dim = int(cfg.get("dim", 0))
        heads = int(cfg.get("heads", 1))
        r = int(cfg.get("sr_ratio", 1))
        if p < 1:
            raise ValueError(f"geron_pvt: stage {i} patch_size must be >= 1, got {p}")
        if dim < 1:
            raise ValueError(f"geron_pvt: stage {i} needs a positive 'dim', got {dim}")
        if heads < 1 or dim % heads != 0:
            raise ValueError(f"geron_pvt: stage {i} dim {dim} is not divisible by {heads} heads")
        if r < 1:
            raise ValueError(f"geron_pvt: stage {i} sr_ratio must be >= 1, got {r}")
        H, W, C = x.shape
        if H % p or W % p:
            raise ValueError(f"geron_pvt: stage {i} patch_size {p} does not divide the {H}x{W} grid")
        gh, gw = H // p, W // p
        fan = p * p * C
        Wm = cfg.get("W")
        if Wm is None:
            Wm = _lcg_matrix(fan, dim, seed + 7919 * i + 1, 1.0 / np.sqrt(fan))
        else:
            Wm = np.atleast_2d(np.asarray(Wm, dtype=float))
            if Wm.shape != (fan, dim):
                raise ValueError(f"geron_pvt: stage {i} W has shape {Wm.shape}, expected {(fan, dim)}")
        patches = x.reshape(gh, p, gw, p, C).transpose(0, 2, 1, 3, 4).reshape(gh, gw, fan)
        x = patches @ Wm
        n_tok = gh * gw
        c = n_tok * max(n_tok // (r * r), 1)
        stages.append(
            {
                "index": i,
                "grid": (int(gh), int(gw)),
                "tokens": int(n_tok),
                "dim": dim,
                "heads": heads,
                "sr_ratio": r,
                "parameters": int(fan * dim),
                "attention_cost": int(c),
                "full_attention_cost": int(n_tok * n_tok),
            }
        )
        params += fan * dim
        cost += c
        full += n_tok * n_tok

    return RichResult(
        title="Pyramid Vision Transformer",
        summary_lines=[("Stages", len(stages)), ("Output", x.shape), ("Attention cost", cost)],
        interpretation="The pyramid restores multi-scale features; spatial reduction pays for the large early grids.",
        payload={
            "tokens": x,
            "stages": stages,
            "output_shape": tuple(int(v) for v in x.shape),
            "n_parameters": int(params),
            "attention_cost": int(cost),
            "full_attention_cost": int(full),
            "estimate": x,
            "n": int(img.size),
            "method": "PVT patch-embedding pyramid executed on the image, with SRA attention costs",
        },
    )


def cheatsheet():
    return "hmpvt: Pyramid Vision Transformer multi-scale stages"
