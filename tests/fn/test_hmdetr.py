"""Tests for hmdetr.geron_detr."""

from morie.fn import _array_core as np

from morie.fn.hmdetr import geron_detr


def test_hmdetr_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    image = rng.normal(0, 1, (3, 224, 224))
    result = geron_detr(image, n_queries=10, n_layers=1, d_model=8, n_heads=2, n_classes=3)
    assert isinstance(result, dict)
    assert result["feature_shape"] == (7, 7)
    assert result["n_tokens"] == 49
    assert result["max_detections"] == 10
    assert result["encoder_attention_cost"] == 2401


def test_hmdetr_edge():
    """Test edge cases."""
    pb = [[0.0, 0.0, 1.0, 1.0], [10.0, 10.0, 11.0, 11.0]]
    pc = [[10.0, 0.0], [0.0, 10.0]]
    image = np.zeros((3, 224, 224))
    result = geron_detr(
        image,
        n_queries=2,
        n_layers=1,
        d_model=8,
        n_heads=2,
        n_classes=2,
        pred_boxes=pb,
        pred_classes=pc,
        gt_boxes=[[0.0, 0.0, 1.0, 1.0]],
        gt_classes=[0],
    )
    assert isinstance(result, dict)
    assert result["matching"] == [(0, 0)]
    assert "loss_bbox" in result
    assert "loss_class" in result
