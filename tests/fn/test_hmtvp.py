"""Tests for hmtvp.geron_torchvision_pretrained."""

import math

import pytest

from morie.fn.hmtvp import geron_torchvision_pretrained


def test_hmtvp_basic():
    """Centre crop to the short side, nearest-neighbour to size, then
    per-channel (x - mean) / sd."""
    img = [[[float(10 * i + j), float(i)] for j in range(6)] for i in range(4)]   # 4 x 6 x 2
    r = geron_torchvision_pretrained(img, 2, mean=[5.0, 1.0], sd=[2.0, 0.5])
    assert (r["size"], r["channels"], r["cropside"]) == (2, 2, 4)
    # crop columns 1..4; samples rows 0, 2 and columns 1, 3
    want = [[[(img[i][j][0] - 5) / 2, (img[i][j][1] - 1) / 0.5] for j in (1, 3)] for i in (0, 2)]
    assert r["pixels"] == want


def test_hmtvp_edge():
    """The read-out is the arg-max of the logits with its softmax weight."""
    img = [[[0.0]]]
    r = geron_torchvision_pretrained(img, 1, [0.0], [1.0], logits=[0.2, 1.5, -0.3], topk=2)
    assert r["pred"] == 1 and r["topk"] == [1, 0]
    z = [math.exp(v) for v in (0.2, 1.5, -0.3)]
    assert r["topprob"] == pytest.approx(z[1] / sum(z), rel=1e-14)
    with pytest.raises(ValueError, match="per channel"):
        geron_torchvision_pretrained(img, 1, [0.0, 0.0], [1.0, 1.0])


