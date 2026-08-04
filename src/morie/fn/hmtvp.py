# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Preprocessing contract for a pretrained image model.

Geron, A. (2026). Hands-On Machine Learning with Scikit-Learn and PyTorch. O'Reilly, ch. 12 pp. 458-459
"""

from . import _geron as _core

from ._richresult import RichResult

__all__ = ["pretprep", "geron_torchvision_pretrained"]

_METHOD = "Preprocessing contract for a pretrained image model"


def pretprep(image, size, mean, sd, logits=None, topk=1):
    """Preprocessing contract for a pretrained image model.

    Preprocess an image for a pretrained model, pp. 458-459.

    The section's substance is the preprocessing contract, not the
    download: the image must be brought to the size the pretrained
    model expects (224 x 224 for ConvNeXt) and the pixel intensities
    standardized per colour channel "using ImageNet's means and
    standard deviations for each channel".  Those constants are NOT
    printed in the extracted text, so ``mean`` and ``sd`` are required
    arguments rather than baked-in defaults.

    ponytail: centre crop to a square, then nearest-neighbour
    subsample.  ``weights.transforms()`` uses bilinear resizing; swap
    the sampler if you need to match torchvision to the last decimal.

    Passing ``logits`` also applies the p. 459 read-out, ``argmax`` over
    the class logits, and reports the top-k classes.

    Parameters
    ----------
    image : as documented for the shelf core
        See ``morie.fn._geron.pretprep``.
    size : as documented for the shelf core
        See ``morie.fn._geron.pretprep``.
    mean : as documented for the shelf core
        See ``morie.fn._geron.pretprep``.
    sd : as documented for the shelf core
        See ``morie.fn._geron.pretprep``.
    logits : as documented for the shelf core
        See ``morie.fn._geron.pretprep``.
    topk : as documented for the shelf core
        See ``morie.fn._geron.pretprep``.

    Returns
    -------
    result : RichResult
        Payload keys: size, channels, cropside, pixelmean.

    References
    ----------
    Geron, A. (2026). Hands-On Machine Learning with Scikit-Learn and PyTorch. O'Reilly, ch. 12 pp. 458-459
    """
    res = _core.pretprep(image=image, size=size, mean=mean, sd=sd, logits=logits, topk=topk)
    return RichResult(
        title=_METHOD,
        summary_lines=[("size", res["size"]), ("channels", res["channels"]), ("cropside", res["cropside"]), ("pixelmean", res["pixelmean"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
geron_torchvision_pretrained = pretprep


def cheatsheet():
    return "pretprep: Preprocessing contract for a pretrained image model"
