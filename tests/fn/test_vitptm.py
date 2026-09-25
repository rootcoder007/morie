"""Tests for vitptm.vit_patch_embed."""

import pytest

from morie.fn.vitptm import vit_patch_embed

IMG = [[1.0, 2.0, 3.0, 4.0],
       [5.0, 6.0, 7.0, 8.0],
       [9.0, 10.0, 11.0, 12.0],
       [13.0, 14.0, 15.0, 16.0]]

# raster order over the 2x2 patch grid; row-then-column inside a patch
PATCHES = [[1.0, 2.0, 5.0, 6.0],
           [3.0, 4.0, 7.0, 8.0],
           [9.0, 10.0, 13.0, 14.0],
           [11.0, 12.0, 15.0, 16.0]]


def test_vitptm_reshape_is_the_flattened_patch_sequence():
    """N = HW/P^2 and x_p in R^{N x (P^2 C)}, Section 3.1 p. 3."""
    out = vit_patch_embed(IMG, 2, 3)
    assert out["n_patches"] == 4
    assert out["estimate"] == 4.0
    assert out["patch_dim"] == 4  # P^2 . C = 4 . 1
    assert out["embed_dim"] == 3
    assert out["grid_rows"] == 2 and out["grid_cols"] == 2
    assert out["n_channels"] == 1
    assert out["patches"] == PATCHES
    # every pixel appears exactly once across the patches
    assert sorted(v for row in out["patches"] for v in row) == [
        float(i) for i in range(1, 17)]


def test_vitptm_embeddings_are_the_patch_times_projection_product():
    out = vit_patch_embed(IMG, 2, 3)
    E = out["projection"]
    assert len(E) == 4 and all(len(r) == 3 for r in E)
    for i, row in enumerate(PATCHES):
        for j in range(3):
            want = sum(row[k] * E[k][j] for k in range(4))
            assert out["embeddings"][i][j] == pytest.approx(want, rel=1e-12)
    assert out["skip_used"] == 12  # skip + P^2 C . D


def test_vitptm_zero_scale_zeroes_the_projection():
    """w_scale = 0 gives E = 0, the degenerate anchor the docstring names;
    the patches themselves are untouched."""
    out = vit_patch_embed(IMG, 2, 3, w_scale=0.0)
    assert out["projection"] == [[0.0] * 3] * 4
    assert out["embeddings"] == [[0.0] * 3] * 4
    assert out["patches"] == PATCHES


def test_vitptm_w_scale_is_linear_in_the_projection():
    one = vit_patch_embed(IMG, 2, 3)
    two = vit_patch_embed(IMG, 2, 3, w_scale=2.0)
    for i in range(4):
        for j in range(3):
            assert two["projection"][i][j] == pytest.approx(
                2.0 * one["projection"][i][j], rel=1e-12)
            assert two["embeddings"][i][j] == pytest.approx(
                2.0 * one["embeddings"][i][j], rel=1e-12)


def test_vitptm_skip_slides_along_the_single_shared_stream():
    """E is read row-major off one deterministic stream, so a skip of one
    shifts the whole matrix by one element."""
    a = [v for row in vit_patch_embed(IMG, 2, 3)["projection"] for v in row]
    b = [v for row in vit_patch_embed(IMG, 2, 3, skip=1)["projection"] for v in row]
    assert a[1:] == pytest.approx(b[:-1], rel=1e-12)
    assert vit_patch_embed(IMG, 2, 3, skip=5)["skip_used"] == 5 + 12


def test_vitptm_channel_major_ordering_for_a_multichannel_image():
    """C = 2: a flattened patch is channel 0's P^2 values then channel 1's."""
    second = [[v + 100.0 for v in row] for row in IMG]
    out = vit_patch_embed([IMG, second], 2, 2)
    assert out["n_channels"] == 2
    assert out["patch_dim"] == 8
    assert out["n_patches"] == 4
    assert out["patches"][0] == [1.0, 2.0, 5.0, 6.0, 101.0, 102.0, 105.0, 106.0]
    assert out["patches"][3] == [11.0, 12.0, 15.0, 16.0, 111.0, 112.0, 115.0, 116.0]


def test_vitptm_one_pixel_patches_are_the_identity_reshape():
    out = vit_patch_embed(IMG, 1, 1)
    assert out["n_patches"] == 16
    assert out["patch_dim"] == 1
    assert out["patches"] == [[float(i)] for i in range(1, 17)]


def test_vitptm_rejects_bad_input():
    with pytest.raises(ValueError, match="divide both H and W"):
        vit_patch_embed(IMG, 3, 2)
    with pytest.raises(ValueError, match="patch_size must be a positive"):
        vit_patch_embed(IMG, 0, 2)
    with pytest.raises(ValueError, match="embed_dim must be a positive"):
        vit_patch_embed(IMG, 2, 0)
