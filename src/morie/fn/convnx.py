# morie.fn -- function file (rootcoder007/morie)
"""ConvNeXt block."""

import math

from . import _array_core as np
from . import _s03core as core
from ._richresult import RichResult

__all__ = ["convnext_block"]


def convnext_block(x, filters=None, kernel=7, expand=4, layer_scale=0.0,
                   seed=42):
    """
    ConvNeXt block

    Formula: depthwise conv + LayerNorm + 1x1 conv

    Depthwise k x k convolution, channel-last LayerNorm, a pointwise
    expansion to 4C with GELU, a pointwise projection back to C, then a
    residual add scaled by gamma.  ConvNeXt initialises the layer scale
    gamma at a small value; at gamma = 0 the block is EXACTLY the
    identity, which is the check that the residual path is wired the
    right way round.

    Parameters
    ----------
    x : array-like
        H x W matrix for a single channel, or a list of such matrices.
    filters : int or None
        Number of channels; inferred from x when None.
    kernel : int
        Depthwise kernel size, odd.
    expand : int
        Pointwise expansion factor.
    layer_scale : float
        Initial gamma of the residual scaling.
    seed : int
        Seed of the deterministic stream for the weights.

    Returns
    -------
    result : dict
        Keys: estimate (mean output), out, residual_norm, H, W, C.

    References
    ----------
    Liu et al. (2022), A ConvNet for the 2020s, CVPR 2022:11976-11986.
    """
    M = core.mat(x)
    H = len(M)
    if H == 0:
        raise ValueError("empty input: x has no rows")
    W = len(M[0])
    C = 1 if filters is None else int(filters)
    if C < 1:
        raise ValueError("filters must be at least 1")
    k = int(kernel)
    if k < 1 or k % 2 == 0:
        raise ValueError("kernel must be odd and positive")
    expand = int(expand)
    if expand < 1:
        raise ValueError("expand must be at least 1")
    rng = np.random.default_rng(seed)
    dw = [[float(rng.normal(0.0, 1.0)) / k for _ in range(k)] for _ in range(k)]
    w1 = [[float(rng.normal(0.0, 0.02)) for _ in range(expand)]
          for _ in range(1)]
    w2 = [[float(rng.normal(0.0, 0.02)) for _ in range(1)]
          for _ in range(expand)]
    r = k // 2
    conv = [[0.0] * W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            s = 0.0
            for a in range(-r, r + 1):
                for b in range(-r, r + 1):
                    ii = min(max(i + a, 0), H - 1)
                    jj = min(max(j + b, 0), W - 1)
                    s += M[ii][jj] * dw[a + r][b + r]
            conv[i][j] = s
    flat = [conv[i][j] for i in range(H) for j in range(W)]
    mu = sum(flat) / len(flat)
    var = sum((v - mu) ** 2 for v in flat) / len(flat)
    inv = 1.0 / math.sqrt(var + 1e-6)
    out = [[0.0] * W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            h = (conv[i][j] - mu) * inv
            acc = 0.0
            for e in range(expand):
                z = h * w1[0][e]
                acc += core.gelu(z) * w2[e][0]
            out[i][j] = M[i][j] + layer_scale * acc
    res = math.sqrt(sum((out[i][j] - M[i][j]) ** 2
                        for i in range(H) for j in range(W)))
    return RichResult(payload={
        "estimate": sum(out[i][j] for i in range(H) for j in range(W)) / (H * W),
        "out": out,
        "residual_norm": res,
        "H": H,
        "W": W,
        "C": C,
        "method": "ConvNeXt block",
    })


def cheatsheet():
    return "convnx: ConvNeXt block"


# compact alias per ledger/NAMING.md
convnextblock = convnext_block
