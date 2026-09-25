"""Tests for unetbk.unet_backbone."""

import pytest

from morie.fn.unetbk import mirror_pad, valid_output_size


def test_unetbk_basic():
    """Ronneberger et al. (2015) Fig. 1: a 572 x 572 tile gives a 388 x 388
    segmentation with depth 4, two valid 3 x 3 convolutions per block;
    the contracting path's pre-pooling sizes are 568, 280, 136, 64."""
    r = valid_output_size(572)
    assert r["output"] == 388 and r["border_lost"] == 184
    assert r["skip_sizes"] == [568, 280, 136, 64]


def test_unetbk_edge():
    """Mirror padding reflects about the edge without repeating it (numpy
    'reflect'); an odd size before a pooling step is rejected."""
    img = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    p = mirror_pad(img, 1)
    assert p[0] == [5.0, 4.0, 5.0, 6.0, 5.0]
    assert p[2] == [5.0, 4.0, 5.0, 6.0, 5.0]
    assert [row[0] for row in p] == [5.0, 2.0, 5.0, 8.0, 5.0]
    with pytest.raises(ValueError, match="odd"):
        valid_output_size(571)
    with pytest.raises(ValueError, match="smaller"):
        mirror_pad(img, 3)


