# morie.fn -- function file (rootcoder007/morie)
r"""SDXL: conditioning on the things the pipeline used to throw away.

Two of SDXL's improvements are not architectural at all. They take
metadata that a latent diffusion pipeline already has and would
otherwise discard, and feed it to the model as conditioning -- costing
no extra supervision and removing two concrete failure modes.

**Size conditioning replaces a choice between two bad options.**
Training needs a minimum image size, so a pipeline either *discards*
everything smaller (Stable Diffusion 1.4/1.5 dropped anything below
512 pixels; at SDXL's 256-pixel pre-training resolution the paper
measures **39% of the dataset** lost) or *upscales* it, which bakes
upscaling artefacts into the model's own outputs. Instead: give the
UNet the **original** height and width as conditioning
:math:`c_{size} = (h_{orig}, w_{orig})`, Fourier-encoded and added to
the timestep embedding. Nothing is discarded, and at inference the
user sets the apparent resolution through the same channel.

**Crop conditioning turns a data augmentation into a control.**
Batching requires equal-sized tensors, so the standard pipeline
resizes the short side and then randomly crops the long one -- and that
random crop *leaks into the samples*, which is why earlier Stable
Diffusion models produce cats with their heads cut off. SDXL samples
the crop coordinates :math:`(c_{top}, c_{left})` and conditions on
them, so the model learns what a crop looks like; setting
:math:`(0, 0)` at inference then asks for an **uncropped** image. The
augmentation is kept and its leakage becomes a knob.

**Multi-aspect training** finetunes over buckets of aspect ratios
holding the pixel count near :math:`1024^2`, because square output is
an unnatural default for landscape and portrait screens.

References
----------
Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T.,
Muller, J., Penna, J. & Rombach, R. (2023) "SDXL: Improving Latent
Diffusion Models for High-Resolution Image Synthesis",
arXiv:2307.01952 (*ICLR 2024*). Sec. 2.1 (a three times larger UNet
backbone, more attention blocks, a second text encoder), Sec. 2.2
("Conditioning the Model on Image Size": the two existing approaches of
discarding images below a minimum resolution or upscaling them, the
measured 39% of data that would be discarded at 256 pixels, and the
proposal to condition on the original height and width, each embedded
by Fourier features, concatenated and ADDED to the timestep embedding;
"Conditioning the Model on Cropping Parameters": random cropping during
training leaking into samples as cut-off objects, uniformly sampling
c_top and c_left and feeding them as Fourier-embedded conditioning, and
setting (0,0) at inference to obtain object-centred samples), Sec. 2.3
(multi-aspect finetuning with buckets keeping the pixel count close to
1024^2), and Sec. 2.5 (the separate refinement model applying a
noising-denoising process to SDXL's latents).

Rombach, R., Blattmann, A., Lorenz, D., Esser, P. & Ommer, B. (2022)
"High-Resolution Image Synthesis with Latent Diffusion Models",
*CVPR 2022*, 10684-10695, arXiv:2112.10752. The latent diffusion
model being improved.

Ho, J., Jain, A. & Abbeel, P. (2020) "Denoising Diffusion
Probabilistic Models", *NeurIPS 2020*, arXiv:2006.11239.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["fourier_embedding", "size_conditioning",
           "crop_conditioning", "discarded_fraction",
           "aspect_ratio_buckets", "condition_vector"]

_EPS = 1e-12


def fourier_embedding(value, dim=8, scale=0.001):
    r"""Fourier feature encoding of one scalar conditioning value."""
    v = float(value)
    n = int(dim)
    if n < 2 or n % 2:
        raise ValueError("sdxlcd: the embedding width must be even "
                         "and at least 2")
    out = []
    for j in range(n // 2):
        f = (2.0 ** j) * math.pi * float(scale)
        out.append(math.sin(f * v))
        out.append(math.cos(f * v))
    return out


def size_conditioning(h_original, w_original, dim=8):
    r""":math:`c_{size} = (h_{orig}, w_{orig})`, before any rescaling.

    Trivially available during training, and it removes the choice
    between discarding small images and upscaling them.
    """
    h, w = float(h_original), float(w_original)
    if h <= 0.0 or w <= 0.0:
        raise ValueError("sdxlcd: the original size must be positive")
    return {"c_size": (h, w),
            "embedding": fourier_embedding(h, dim)
            + fourier_embedding(w, dim),
            "note": "the ORIGINAL size, so no training image has to "
                    "be thrown away or upscaled"}


def crop_conditioning(c_top=0, c_left=0, dim=8):
    r""":math:`c_{crop} = (c_{top}, c_{left})`, in pixels.

    At inference :math:`(0,0)` asks for an image that was not cropped
    -- the augmentation is kept but no longer leaks.
    """
    t, l = float(c_top), float(c_left)
    if t < 0.0 or l < 0.0:
        raise ValueError("sdxlcd: crop offsets cannot be negative")
    return {"c_crop": (t, l),
            "embedding": fourier_embedding(t, dim)
            + fourier_embedding(l, dim),
            "object_centred": t == 0.0 and l == 0.0,
            "note": "(0,0) at inference asks for an UNCROPPED image"}


def sample_crop(height, width, target_h, target_w, rng):
    r"""The random crop the pipeline performs anyway."""
    H, W = int(height), int(width)
    th, tw = int(target_h), int(target_w)
    if th > H or tw > W:
        raise ValueError("sdxlcd: the target is larger than the "
                         "image")
    t = int(float(rng.uniform()) * (H - th + 1))
    l = int(float(rng.uniform()) * (W - tw + 1))
    return {"c_top": min(t, H - th), "c_left": min(l, W - tw)}


def discarded_fraction(sizes, minimum=256):
    r"""How much of the dataset a resolution filter throws away.

    The paper measures 39% at its 256-pixel pre-training resolution;
    with size conditioning the answer is 0.
    """
    S = [(float(a), float(b)) for a, b in sizes]
    if not S:
        raise ValueError("sdxlcd: no image sizes given")
    m = float(minimum)
    lost = sum(1 for h, w in S if h < m or w < m)
    return {"discarded": lost, "total": len(S),
            "fraction": lost / float(len(S)),
            "kept_with_conditioning": len(S),
            "minimum": m,
            "note": "conditioning keeps every image; filtering does "
                    "not"}


def aspect_ratio_buckets(ratios, pixels=1024 * 1024, multiple=64):
    r"""Buckets of differing aspect ratio at near-constant pixel
    count."""
    out = []
    for r in ratios:
        a = float(r)
        if a <= 0.0:
            raise ValueError("sdxlcd: an aspect ratio must be "
                             "positive")
        h = math.sqrt(float(pixels) / a)
        w = a * h
        M = int(multiple)
        hh = max(M, int(round(h / M)) * M)
        ww = max(M, int(round(w / M)) * M)
        out.append({"aspect": a, "height": hh, "width": ww,
                    "pixels": hh * ww,
                    "pixel_error": abs(hh * ww - pixels)
                    / float(pixels)})
    return {"buckets": out,
            "max_pixel_error": max(b["pixel_error"] for b in out),
            "note": "square output is an unnatural default for "
                    "landscape and portrait screens"}


def condition_vector(h_original, w_original, c_top=0, c_left=0,
                     timestep_embedding=None, dim=8):
    r"""Concatenate the conditionings, then ADD to the timestep
    embedding."""
    s = size_conditioning(h_original, w_original, dim)
    c = crop_conditioning(c_top, c_left, dim)
    cat = s["embedding"] + c["embedding"]
    if timestep_embedding is None:
        vec = cat
    else:
        t = [float(v) for v in k.vec(timestep_embedding)]
        if len(t) != len(cat):
            raise ValueError("sdxlcd: the timestep embedding is %d "
                             "wide but the conditioning is %d"
                             % (len(t), len(cat)))
        vec = [t[i] + cat[i] for i in range(len(cat))]
    return RichResult(payload={
        "estimate": vec, "vector": vec, "width": len(vec),
        "c_size": s["c_size"], "c_crop": c["c_crop"],
        "method": "SDXL micro-conditioning; Podell et al. (2023)",
        "note": "concatenated, then ADDED to the timestep embedding "
                "in the UNet",
    })


def cheatsheet():
    return ("sdxlcd: two improvements that add NO supervision -- they "
            "condition on metadata the pipeline already had and threw "
            "away. SIZE: filtering below a minimum resolution "
            "discarded 39% of the data and upscaling bakes in "
            "artefacts, so give the UNet the ORIGINAL (h,w) as "
            "Fourier-embedded conditioning added to the timestep "
            "embedding. CROP: batching forces a random crop that LEAKS "
            "into samples (cut-off heads), so condition on "
            "(c_top,c_left) and set (0,0) at inference to ask for an "
            "uncropped image. Plus multi-aspect buckets at ~1024^2 "
            "pixels.")


# compact alias per ledger/NAMING.md
sdxlconditioning = condition_vector

# public names resolved by fn/_lazy_map.json
sdxl_unet = condition_vector
sdxlunet = condition_vector
