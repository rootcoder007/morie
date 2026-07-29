# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Image segmentation via k-means on pixel colors."""

import numpy as np

from ._richresult import RichResult
from .hmkmn import geron_kmeans

__all__ = ["geron_image_segmentation"]

_METHOD = "Colour segmentation by k-means on pixels"


def geron_image_segmentation(image, n_clusters, seed=0):
    """
    Image segmentation via k-means on pixel colors.

    Formula: cluster pixels in RGB space; replace with cluster mean

    Colour segmentation, not semantic segmentation: pixels are clustered
    on colour alone with no spatial term at all, so two unrelated
    regions of the same shade land in one segment.  That is the whole
    limitation of the technique and it is why the returned segment map
    is not a set of connected components.

    The clustering itself is delegated to
    :func:`morie.fn.hmkmn.geron_kmeans`; this reshapes ``(h, w, c)`` to
    ``(h*w, c)`` and back, and reports the compression achieved --
    the segmented image needs only ``n_clusters`` colours, which is the
    same computation as colour quantisation.

    Parameters
    ----------
    image : array-like, shape (h, w, c) or (h, w)
        Pixel array.  A 2-D input is treated as single-channel.
    n_clusters : int
        Number of colour segments.
    seed : int
        Seed passed through to k-means.

    Returns
    -------
    result : RichResult
        Keys: segmented, labels, palette, inertia, compression_ratio,
        estimate, n, method.

    Examples
    --------
    A 2x2 image with two black and two white pixels splits cleanly, and
    every pixel is replaced by its exact segment colour:

    >>> img = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ...        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
    >>> r = geron_image_segmentation(img, n_clusters=2, seed=0)
    >>> r["segmented"].shape
    (2, 2, 3)
    >>> round(r["inertia"], 12)
    0.0
    >>> sorted(round(float(v), 6) for v in r["palette"].ravel())
    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    Nearby shades merge into one segment; the mean of 0.0 and 0.1 is
    0.05:

    >>> g = geron_image_segmentation([[0.0, 0.1, 0.9]], n_clusters=2, seed=0)
    >>> sorted(round(float(v), 6) for v in g["palette"].ravel())
    [0.05, 0.9]

    References
    ----------
    Géron Ch 8
    """
    img = np.asarray(image, dtype=float)
    if img.ndim == 2:
        img = img[:, :, None]
    if img.ndim == 1:
        img = img.reshape(1, -1, 1)
    if img.ndim != 3:
        raise ValueError(f"geron_image_segmentation: image must be 2-D or 3-D (h, w, c), got ndim={img.ndim}")
    if img.size == 0:
        raise ValueError("geron_image_segmentation: image is empty")
    if not np.all(np.isfinite(img)):
        raise ValueError("geron_image_segmentation: image contains non-finite values")

    h, w, c = img.shape
    flat = img.reshape(h * w, c)
    k = int(n_clusters)
    if not (1 <= k <= h * w):
        raise ValueError(f"geron_image_segmentation: n_clusters must lie in 1..{h * w}, got {n_clusters!r}")

    km = geron_kmeans(flat, n_clusters=k, seed=int(seed))
    labels = km["labels"]
    palette = km["centers"]
    seg = palette[labels].reshape(h, w, c)

    distinct = int(np.unique(flat, axis=0).shape[0])
    ratio = float(distinct) / k

    return RichResult(
        title="Colour segmentation",
        summary_lines=[
            ("Image", f"{h} x {w} x {c}"),
            ("Segments", k),
            ("Distinct colours before", distinct),
            ("Inertia", float(km["inertia"])),
        ],
        interpretation=(
            "Colour-only segmentation: no spatial term, so disconnected regions of the same shade "
            "share a segment."
        ),
        payload={
            "segmented": seg,
            "labels": labels.reshape(h, w),
            "palette": palette,
            "inertia": float(km["inertia"]),
            "compression_ratio": ratio,
            "estimate": float(km["inertia"]),
            "n": int(h * w),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmiseg: colour segmentation -- k-means over pixels (delegates to hmkmn), palette replacement"
