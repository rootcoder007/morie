"""Tests for alocp.alammar_openclip_contrastive."""

from morie.fn.alocp import alammar_openclip_contrastive


def test_alocp_basic():
    I = [[1.0, 0.0], [0.0, 1.0]]
    out = alammar_openclip_contrastive(I, I, tau=0.5)
    assert out["image_to_text_loss"] == out["text_to_image_loss"]


def test_alocp_edge():
    import pytest
    with pytest.raises(ValueError, match="at least 2"):
        alammar_openclip_contrastive([[1.0]], [[1.0]])
