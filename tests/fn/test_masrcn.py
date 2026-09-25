"""Tests for masrcn.mask_rcnn_segmentation (RoIAlign, He et al. 2017)."""

import pytest

from morie.fn.masrcn import alignment_error, mask_rcnn_segmentation


def _plane(h=8, w=9):
    # F[y][x] = 2y - 0.5x + 3: bilinear interpolation reproduces it exactly
    return [[2.0 * y - 0.5 * x + 3.0 for x in range(w)] for y in range(h)]


def test_masrcn_basic():
    """On a linear feature map every bilinear sample is exact and the
    samples are symmetric about the bin centre, so each RoIAlign bin
    equals the plane at its centre: y0 + bh (i + 1/2), x0 + bw (j + 1/2),
    with the box divided by the stride and never rounded."""
    box, stride, n = (2.4, 1.2, 12.8, 13.6), 2.0, 3
    result = mask_rcnn_segmentation(_plane(), box, out_size=n, stride=stride, samples=2)
    assert isinstance(result, dict)
    y0, x0, y1, x1 = [v / stride for v in box]
    bh, bw = (y1 - y0) / n, (x1 - x0) / n
    for i in range(n):
        for j in range(n):
            yc, xc = y0 + bh * (i + 0.5), x0 + bw * (j + 0.5)
            assert result["pooled"][i][j] == pytest.approx(2 * yc - 0.5 * xc + 3, rel=1e-13)
    assert result["exact_box"] == pytest.approx((y0, x0, y1, x1), rel=1e-15)


def test_masrcn_edge():
    """RoIPool's quantisation shift is the fractional part of the scaled
    box corner, and it multiplies by the stride in the image; a box with
    no extent is refused."""
    e = alignment_error(_plane(), (2.4, 1.2, 12.8, 13.6), stride=2.0)
    assert e["feature_shift"] == pytest.approx((0.2, 0.6), rel=1e-12)
    assert e["input_pixel_shift"] == pytest.approx((0.4, 1.2), rel=1e-12)
    with pytest.raises(ValueError):
        mask_rcnn_segmentation(_plane(), (3.0, 3.0, 3.0, 5.0))
