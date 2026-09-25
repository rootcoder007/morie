"""Tests for ocrwit.ocr_wit_layout (LayoutLMv3 word-patch alignment)."""

import math

import pytest

from morie.fn.ocrwit import ocr_wit_layout, patch_of_box

W, H, G = 280.0, 140.0, 14   # patches are 20 x 10 page units


def _cells(box):
    x0, y0, x1, y1 = box
    cs = range(int(x0 // 20), max(int(math.ceil(x1 / 20)) - 1, int(x0 // 20)) + 1)
    rs = range(int(y0 // 10), max(int(math.ceil(y1 / 10)) - 1, int(y0 // 10)) + 1)
    return sorted(r * G + c for r in rs for c in cs)


BOXES = [(12.0, 3.0, 18.0, 8.0),      # right half of patch (0, 0)
         (35.0, 12.0, 65.0, 18.0),    # row 1, columns 1..3
         (100.0, 50.0, 100.0, 50.0),  # a point on the corner of cells
         (270.0, 131.0, 280.0, 140.0)]


def test_ocrwit_basic():
    """Each word covers exactly the grid cells its box overlaps (computed
    here from the 20 x 10 patch size); a word's label is 1 when any of
    those patches is masked, and masked words are left out."""
    masked = [0, 17]
    r = ocr_wit_layout(BOXES, masked, W, H, patch_grid=G, masked_text=[3])
    assert isinstance(r, dict)
    for i in range(3):
        assert r["patches"][i] == _cells(BOXES[i])
        assert r["labels"][i] == int(any(p in masked for p in _cells(BOXES[i])))
    assert 3 not in r["labels"]
    assert r["labels"] == {0: 1, 1: 1, 2: 0}


def test_ocrwit_edge():
    """A box in the right half of the first patch stays in patch 0;
    masking every word leaves no examples."""
    assert patch_of_box((12.0, 3.0, 18.0, 8.0), W, H, G) == [0]
    with pytest.raises(ValueError):
        ocr_wit_layout(BOXES[:1], [0], W, H, masked_text=[0])
