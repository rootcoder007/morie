# morie.fn -- function file (rootcoder007/morie)
"""ControlNet attachment with zero convolutions."""

import math

from . import _array_core as np
from . import _s03core as core
from ._richresult import RichResult

__all__ = ["controlnet_attach"]


def controlnet_attach(base, condition, zero_conv_weight=0.0, seed=42):
    """
    ControlNet attachment

    Formula: trainable copy of the UNet encoder; zero-conv

    The conditioning branch is a copy of the frozen encoder whose output
    passes through a 1x1 convolution initialised at ZERO before it is
    added back to the base feature map.  At initialisation the sum is
    therefore exactly the base output: attaching a ControlNet cannot
    perturb the pretrained model until the zero-convolution has learned
    a non-zero weight.  That identity is the whole design and is the
    anchor here.

    Parameters
    ----------
    base : array-like
        Base-network feature map, H x W.
    condition : array-like
        Conditioning image of the same shape.
    zero_conv_weight : float
        Weight of the zero convolution; 0 at initialisation.
    seed : int
        Seed of the deterministic stream for the trainable copy.

    Returns
    -------
    result : dict
        Keys: estimate (mean output), out, delta_norm, control,
        is_identity, H, W.

    References
    ----------
    Zhang, Rao & Agrawala (2023), Adding Conditional Control to
    Text-to-Image Diffusion Models, ICCV 2023:3836-3847.
    """
    B = core.mat(base)
    Cm = core.mat(condition)
    H = len(B)
    if H == 0:
        raise ValueError("empty input: base has no rows")
    W = len(B[0])
    if len(Cm) != H or len(Cm[0]) != W:
        raise ValueError("base and condition must have the same shape")
    rng = np.random.default_rng(seed)
    w = [[float(rng.normal(0.0, 0.5)) for _ in range(3)] for _ in range(3)]
    ctrl = [[0.0] * W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            s = 0.0
            for a in range(-1, 2):
                for b in range(-1, 2):
                    ii = min(max(i + a, 0), H - 1)
                    jj = min(max(j + b, 0), W - 1)
                    s += Cm[ii][jj] * w[a + 1][b + 1]
            ctrl[i][j] = core.gelu(s)
    out = [[B[i][j] + zero_conv_weight * ctrl[i][j] for j in range(W)]
           for i in range(H)]
    dn = math.sqrt(sum((out[i][j] - B[i][j]) ** 2
                       for i in range(H) for j in range(W)))
    return RichResult(payload={
        "estimate": sum(out[i][j] for i in range(H) for j in range(W)) / (H * W),
        "out": out,
        "control": ctrl,
        "delta_norm": dn,
        "is_identity": 1 if dn == 0.0 else 0,
        "H": H,
        "W": W,
        "method": "ControlNet attachment with a zero convolution",
    })


def cheatsheet():
    return "cncpat: ControlNet attachment with a zero convolution"


# compact alias per ledger/NAMING.md
controlnetattach = controlnet_attach
