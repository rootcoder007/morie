"""Tests for alcap.alammar_image_captioning_pipeline."""

from morie.fn.alcap import alammar_image_captioning_pipeline


def test_alcap_basic():
    out = alammar_image_captioning_pipeline("img", lambda im: [1.0, 2.0],
        [[1.0, 0.0]], lambda z, p: "cap")
    assert out["projected"] == [1.0]


def test_alcap_edge():
    import pytest
    with pytest.raises(ValueError, match="callable"):
        alammar_image_captioning_pipeline("img", None, [[1.0]], None)
