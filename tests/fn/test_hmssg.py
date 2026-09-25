"""Tests for hmssg.geron_semantic_segmentation."""

import math

from morie.fn import _array_core as np

from morie.fn.hmssg import geron_semantic_segmentation


def test_hmssg_basic():
    """Test basic functionality without ground truth."""
    rng = np.random.default_rng(42)
    H, W, K = 4, 1, 3
    image = rng.normal(0, 1, (H, W))

    def model(img):
        return rng.normal(0, 1, (H, W, K))

    result = geron_semantic_segmentation(image, model)
    assert isinstance(result, dict)
    assert "labels" in result
    assert "scores" in result
    assert "class_counts" in result
    # argmax over the class axis of an (H, W, K) score map is (H, W)
    assert result["labels"].shape == (H, W)
    assert result["scores"].shape == (H, W, K)
    counts_total = 0
    for c in result["class_counts"]:
        counts_total += int(c)
    assert counts_total == H * W


def test_hmssg_edge():
    """Test functionality with ground truth evaluation metrics."""
    rng = np.random.default_rng(42)
    H, W, K = 4, 1, 3
    image = rng.normal(0, 1, (H, W))

    def model(img):
        return rng.normal(0, 1, (H, W, K))

    y_true = rng.integers(0, K, (H, W))
    result = geron_semantic_segmentation(image, model, y_true=y_true)
    assert isinstance(result, dict)
    assert "labels" in result
    assert "pixel_accuracy" in result
    assert "mean_iou" in result
    assert "iou" in result
    assert "confusion" in result
    pa = float(result["pixel_accuracy"])
    miou = float(result["mean_iou"])
    assert math.isfinite(pa)
    assert 0.0 <= pa <= 1.0
    assert math.isfinite(miou)
    assert 0.0 <= miou <= 1.0
    assert len(result["iou"]) == K
