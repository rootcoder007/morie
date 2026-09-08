# morie.fn -- function file (rootcoder007/morie)
r"""DiT: diffusion with a transformer instead of a U-Net.

Image generative models were the holdout as transformers took over
every other domain: diffusion models used a convolutional U-Net
backbone essentially by default. DiT replaces it with a transformer
operating on **latent patches**, and then asks the question that makes
the paper: does the standard scaling story hold for diffusion?

**Complexity is measured in Gflops, not parameters.** That choice
matters, because a transformer's forward-pass cost can be raised three
ways -- depth, width, or *number of input tokens* -- and the last
changes no parameters at all. Measuring in parameters would make the
patch-size axis invisible. The finding is that DiTs with higher Gflops
consistently have lower FID, whichever axis supplied them.

**Patch size is the cheapest and most brutal knob.** Patchifying a
:math:`I \times I` latent with patch :math:`p` gives
:math:`(I/p)^2` tokens, so halving :math:`p` **quadruples** the token
count and roughly quadruples the Gflops -- with the parameter count
essentially unchanged. That is why DiT-XL/2 (patch 2) is the strong
model and why ``patch_grid`` reports both counts side by side.

**Conditioning by adaLN-zero.** Timestep and class are injected by
regressing the LayerNorm scale and shift -- and additionally a residual
scaling :math:`\alpha` -- from the conditioning vector, with
:math:`\alpha` **initialised to zero** so every DiT block starts as the
identity function. Ablations find this beats in-context conditioning
and cross-attention. The zero-initialisation is the part that is easy
to omit and does the work: it makes the initial network a clean
residual path.

References
----------
Peebles, W. & Xie, S. (2023) "Scalable Diffusion Models with
Transformers", *Proceedings of the IEEE/CVF International Conference
on Computer Vision (ICCV 2023)*, 4195-4205,
doi:10.1109/ICCV51070.2023.00387, arXiv:2212.09748. The abstract:
latent diffusion models with the commonly-used U-Net backbone replaced
by a transformer operating on latent patches; scalability analysed
through forward pass complexity measured in Gflops; DiTs with higher
Gflops -- through increased depth, width, or number of input tokens --
consistently having lower FID; and DiT-XL/2 outperforming all prior
diffusion models on class-conditional ImageNet 512x512 and 256x256
with a state-of-the-art FID of 2.27. Sec. 3 (patchify and the token
count, and the adaLN-zero conditioning block with zero-initialised
residual scaling).

Rombach, R., Blattmann, A., Lorenz, D., Esser, P. & Ommer, B. (2022)
"High-Resolution Image Synthesis with Latent Diffusion Models",
*CVPR 2022*, 10684-10695, doi:10.1109/CVPR52688.2022.01042. The latent
diffusion framework DiT is built inside.

Ho, J., Jain, A. & Abbeel, P. (2020) "Denoising Diffusion
Probabilistic Models", *NeurIPS 2020*, 6840-6851,
arXiv:2006.11239. The diffusion process itself.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["patch_grid", "gflops", "adaln_zero", "dit_block",
           "scaling_comparison"]

_EPS = 1e-12


def patch_grid(latent_size, patch):
    r"""Tokens from patchifying: :math:`T = (I/p)^2`.

    Halving the patch quadruples the tokens and leaves the parameter
    count essentially untouched -- the axis a parameter count cannot
    see.
    """
    I, p = int(latent_size), int(patch)
    if p < 1 or I < 1:
        raise ValueError("dits16: the latent size and patch must be "
                         "positive")
    if I % p != 0:
        raise ValueError("dits16: the patch %d does not divide the "
                         "latent size %d" % (p, I))
    t = (I // p) ** 2
    return {"tokens": t, "grid": I // p, "patch": p,
            "latent_size": I,
            "note": "the token count scales as 1/p^2, at constant "
                    "parameters"}


def gflops(tokens, depth, width, mlp_ratio=4.0):
    r"""Forward-pass cost, the complexity measure the paper uses.

    Attention contributes :math:`O(T^2 d)` and the MLP
    :math:`O(T d^2)`, so the token count enters quadratically -- which
    is why the patch axis moves Gflops so much.
    """
    T, L, d = int(tokens), int(depth), int(width)
    if min(T, L, d) < 1:
        raise ValueError("dits16: tokens, depth and width must be "
                         "positive")
    attn = 4.0 * T * d * d + 2.0 * T * T * d
    mlp = 2.0 * float(mlp_ratio) * T * d * d
    return {"gflops": L * (attn + mlp) * 2.0 / 1e9,
            "tokens": T, "depth": L, "width": d,
            "attention_share": attn / (attn + mlp),
            "note": "measured in Gflops, not parameters, so the "
                    "token axis is visible"}


def adaln_zero(cond, hidden, W_scale, W_shift, W_alpha, eps=1e-6):
    r"""Adaptive LayerNorm with a ZERO-initialised residual gate.

    Regresses :math:`\gamma`, :math:`\beta` and :math:`\alpha` from
    the conditioning vector. With :math:`W_\alpha = 0` every block is
    the identity at initialisation, which is the part that is easy to
    drop and does the work.
    """
    c = [float(v) for v in k.vec(cond)]
    h = [float(v) for v in k.vec(hidden)]
    d = len(h)

    def reg(W):
        if len(W) != d:
            raise ValueError("dits16: the conditioning projection is "
                             "mis-sized (%d rows for %d channels)"
                             % (len(W), d))
        return [sum(W[o][j] * c[j] for j in range(len(c)))
                for o in range(d)]

    m = sum(h) / d
    s = math.sqrt(sum((v - m) ** 2 for v in h) / d + float(eps))
    norm = [(v - m) / s for v in h]
    g, b, a = reg(W_scale), reg(W_shift), reg(W_alpha)
    mod = [norm[i] * (1.0 + g[i]) + b[i] for i in range(d)]
    return {"modulated": mod, "gate": a,
            "identity_at_init": all(abs(v) < 1e-12 for v in a),
            "note": "alpha initialised to ZERO makes the block the "
                    "identity, leaving a clean residual path"}


def dit_block(hidden, cond, attn_fn, mlp_fn, W_scale, W_shift,
              W_alpha, W_scale2, W_shift2, W_alpha2):
    r"""One DiT block: adaLN-zero, attention, adaLN-zero, MLP."""
    h = [float(v) for v in k.vec(hidden)]
    a1 = adaln_zero(cond, h, W_scale, W_shift, W_alpha)
    att = [float(v) for v in attn_fn(a1["modulated"])]
    h = [h[i] + a1["gate"][i] * att[i] for i in range(len(h))]
    a2 = adaln_zero(cond, h, W_scale2, W_shift2, W_alpha2)
    ml = [float(v) for v in mlp_fn(a2["modulated"])]
    h = [h[i] + a2["gate"][i] * ml[i] for i in range(len(h))]
    return {"output": h,
            "identity_at_init": a1["identity_at_init"]
            and a2["identity_at_init"]}


def scaling_comparison(configs):
    r"""Rank configurations by Gflops.

    Each entry is ``(name, latent, patch, depth, width)``. The point
    is that three different axes land on the same scale.
    """
    out = []
    for (name, I, p, L, d) in configs:
        t = patch_grid(I, p)["tokens"]
        g = gflops(t, L, d)
        params = L * (4 * d * d + 8 * d * d)
        out.append({"name": name, "tokens": t,
                    "gflops": g["gflops"], "parameters": params})
    out.sort(key=lambda r: r["gflops"])
    return {"ranked": out,
            "note": "depth, width and TOKEN COUNT all move Gflops; "
                    "only the first two move parameters"}


def cheatsheet():
    return ("dits16: replace the diffusion U-Net with a TRANSFORMER on "
            "latent patches. Complexity is measured in GFLOPS, not "
            "parameters -- deliberately, because raising the token "
            "count changes cost without changing parameters, and "
            "higher Gflops means lower FID whichever axis supplied "
            "them. Halving the patch QUADRUPLES the tokens at roughly "
            "constant parameters, which is why DiT-XL/2 is the strong "
            "model. Conditioning by adaLN-ZERO: regress the norm "
            "scale, shift and a residual gate from the condition, with "
            "the gate initialised to ZERO so each block starts as the "
            "identity.")


# compact alias per ledger/NAMING.md
diffusiontransformer = dit_block

# public names resolved by fn/_lazy_map.json
dit_diffusion_transformer = dit_block
