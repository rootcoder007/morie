# morie.fn -- function file (rootcoder007/morie)
r"""U-Net: segmentation from very few annotated images.

The premise of the paper is a constraint, not an architecture:
biomedical segmentation has thousands of pixels of supervision but
almost no annotated *images*, so the network and the training strategy
have to make heavy use of data augmentation and of every pixel that
exists.

**The shape.** A **contracting path** captures context by pooling; a
symmetric **expanding path** restores resolution, with pooling replaced
by upsampling. High-resolution features from the contracting path are
concatenated into the upsampled ones -- the **skip connections** --
because localisation needs detail that pooling destroyed, and context
alone cannot supply it. The expansive path keeps a large number of
feature channels so context propagates to the high-resolution layers,
which is what makes it *symmetric* to the contracting path and gives
the u-shape.

**No fully connected layers, and only valid convolutions.** So the
segmentation map contains exactly those pixels for which the full
context is available in the input -- the output is smaller than the
input by construction, and the arithmetic of that shrinkage is
something an implementation gets right or silently mis-crops.
``valid_output_size`` computes it.

**The overlap-tile strategy** segments arbitrarily large images: to
predict a tile, the input includes a border of context around it, and
missing data at the image edge is extrapolated by **mirroring**. This
is what lets a fixed-size network handle any image without seams.

**Weighted loss for touching objects.** Where instances of the same
class touch, the separating background must be learned, so a
precomputed **weight map** raises the loss on those narrow borders --
otherwise the network merges adjacent cells and the pixel accuracy
barely notices.

References
----------
Ronneberger, O., Fischer, P. & Brox, T. (2015) "U-Net: Convolutional
Networks for Biomedical Image Segmentation", *Medical Image Computing
and Computer-Assisted Intervention (MICCAI 2015)*, LNCS 9351,
234-241, doi:10.1007/978-3-319-24574-4_28, arXiv:1505.04597. The
abstract (training deep networks is thought to need many thousand
annotated samples; the strategy relies on strong data augmentation; a
contracting path to capture context and a symmetric expanding path
enabling precise localization; end-to-end training from very few
images; winning the ISBI cell tracking challenge 2015; segmentation of
a 512x512 image in under a second). Sec. 2 (pooling operators replaced
by upsampling; high-resolution features from the contracting path
combined with the upsampled output; a large number of feature channels
in the expansive path propagating context to higher resolution layers,
making it symmetric and yielding the u-shape; no fully connected
layers and only the valid part of each convolution, so the map
contains only pixels with full context). Figure 2 and Sec. 3 (the
overlap-tile strategy with missing input extrapolated by mirroring,
and the weight map for separating touching objects).

Long, J., Shelhamer, E. & Darrell, T. (2015) "Fully convolutional
networks for semantic segmentation", *CVPR 2015*, 3431-3440,
doi:10.1109/CVPR.2015.7298965. The fully convolutional predecessor.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["valid_output_size", "mirror_pad", "overlap_tiles",
           "skip_concat", "separation_weight_map"]

_EPS = 1e-12


def valid_output_size(input_size, depth=4, convs_per_block=2,
                      kernel=3):
    r"""Output size after a U-Net of the given depth, valid
    convolutions only.

    Each convolution removes ``kernel - 1`` pixels; each pooling halves
    and each upsampling doubles. The output is *smaller* than the
    input, which is the property that makes tiling necessary.
    """
    s = int(input_size)
    kk = int(kernel) - 1
    if s < 1 or int(depth) < 0:
        raise ValueError("unetbk: the input size must be positive and "
                         "the depth non-negative")
    sizes = []
    for _ in range(int(depth)):
        s -= kk * int(convs_per_block)
        if s < 1:
            raise ValueError("unetbk: the input is too small for this "
                             "depth")
        sizes.append(s)
        if s % 2 != 0:
            raise ValueError("unetbk: size %d is odd before pooling; "
                             "U-Net requires even sizes at every "
                             "pooling step" % s)
        s //= 2
    s -= kk * int(convs_per_block)
    for d in range(int(depth) - 1, -1, -1):
        s *= 2
        s = min(s, sizes[d])
        s -= kk * int(convs_per_block)
        if s < 1:
            raise ValueError("unetbk: the expansive path ran out of "
                             "size")
    return {"output": s, "input": int(input_size),
            "border_lost": int(input_size) - s,
            "skip_sizes": sizes,
            "note": "valid convolutions only, so the output covers "
                    "only pixels with full context"}


def mirror_pad(image, pad):
    r"""Extrapolate missing border data by mirroring.

    Padding with zeros would invent black tissue at the edge; mirroring
    invents plausible tissue, which is why the paper specifies it.
    """
    img = [[float(v) for v in r] for r in k.mat(image)]
    p = int(pad)
    if p < 0:
        raise ValueError("unetbk: the pad must be non-negative")
    h, w = len(img), len(img[0])
    if p >= h or p >= w:
        raise ValueError("unetbk: the mirror pad (%d) must be smaller "
                         "than the image (%dx%d)" % (p, h, w))
    out = []
    for i in range(-p, h + p):
        ii = -i if i < 0 else (2 * h - 2 - i if i >= h else i)
        row = []
        for j in range(-p, w + p):
            jj = -j if j < 0 else (2 * w - 2 - j if j >= w else j)
            row.append(img[ii][jj])
        out.append(row)
    return out


def overlap_tiles(height, width, tile, border):
    r"""Tile an image so every output pixel is predicted once.

    Tiles overlap in the *input* by the border, and their outputs
    abut -- which is what makes the segmentation seamless.
    """
    t, b = int(tile), int(border)
    if t < 1 or b < 0:
        raise ValueError("unetbk: the tile must be positive and the "
                         "border non-negative")
    out = t - 2 * b
    if out < 1:
        raise ValueError("unetbk: the border consumes the whole tile")
    tiles = []
    for i in range(0, int(height), out):
        for j in range(0, int(width), out):
            tiles.append({"output_origin": (i, j),
                          "input_origin": (i - b, j - b),
                          "input_size": t, "output_size": out})
    return {"tiles": tiles, "n_tiles": len(tiles),
            "output_size": out,
            "note": "inputs overlap by the border; outputs abut, so "
                    "there are no seams"}


def skip_concat(upsampled, contracting):
    r"""Concatenate the contracting-path features, centre-cropped.

    The contracting feature map is larger because of the valid
    convolutions, so it is cropped to the upsampled size -- getting
    this crop wrong misaligns every skip connection.
    """
    up = [[float(v) for v in r] for r in k.mat(upsampled)]
    co = [[float(v) for v in r] for r in k.mat(contracting)]
    hu, wu = len(up), len(up[0])
    hc, wc = len(co), len(co[0])
    if hc < hu or wc < wu:
        raise ValueError("unetbk: the contracting map (%dx%d) is "
                         "smaller than the upsampled one (%dx%d)"
                         % (hc, wc, hu, wu))
    oi, oj = (hc - hu) // 2, (wc - wu) // 2
    crop = [[co[oi + i][oj + j] for j in range(wu)]
            for i in range(hu)]
    return {"concatenated": [up[i] + crop[i] for i in range(hu)],
            "crop_offset": (oi, oj), "channels": 2,
            "note": "localisation needs the detail pooling destroyed; "
                    "context alone cannot supply it"}


def separation_weight_map(labels, w0=10.0, sigma=5.0):
    r"""Raise the loss on the thin background between touching
    objects.

    :math:`w(x) = w_c(x) + w_0\exp(-(d_1+d_2)^2/2\sigma^2)`, with
    :math:`d_1, d_2` the distances to the two nearest instances.
    Without it the network merges adjacent cells and pixel accuracy
    barely moves.
    """
    lab = [[int(v) for v in r] for r in k.mat(labels)]
    h, w = len(lab), len(lab[0])
    ids = sorted(set(v for r in lab for v in r if v > 0))
    out = []
    for i in range(h):
        row = []
        for j in range(w):
            if lab[i][j] > 0:
                row.append(1.0)
                continue
            ds = []
            for idv in ids:
                best = None
                for a in range(h):
                    for b in range(w):
                        if lab[a][b] == idv:
                            d = math.sqrt((a - i) ** 2 + (b - j) ** 2)
                            best = d if best is None else min(best, d)
                if best is not None:
                    ds.append(best)
            ds.sort()
            if len(ds) >= 2:
                row.append(1.0 + float(w0) * math.exp(
                    -((ds[0] + ds[1]) ** 2)
                    / (2.0 * float(sigma) ** 2)))
            else:
                row.append(1.0)
        out.append(row)
    return {"weights": out, "n_instances": len(ids),
            "max_weight": max(v for r in out for v in r),
            "note": "the separating background must be LEARNED, so it "
                    "is weighted up"}


def cheatsheet():
    return ("unetbk: built for the case where annotated IMAGES are "
            "scarce though pixels are plentiful. Contracting path for "
            "context, symmetric expanding path for localisation, and "
            "SKIP CONNECTIONS carrying high-resolution detail that "
            "pooling destroyed -- context alone cannot localise. Only "
            "VALID convolutions and no fully connected layers, so the "
            "output is smaller than the input and covers only pixels "
            "with full context; hence the OVERLAP-TILE strategy with "
            "missing border data MIRRORED. A weight map raises the "
            "loss on the thin background between touching objects.")


# compact alias per ledger/NAMING.md
unet = valid_output_size
