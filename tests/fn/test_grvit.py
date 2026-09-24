"""Tests for grvit.geron_vit_patch_embedding."""

from morie.fn import _array_core as np

from morie.fn.grvit import geron_vit_patch_embedding


def test_grvit_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # Image (H, W, C) = (8, 8, 3) with patch_size 2 -> 16 patches
    H, W, C = 8, 8, 3
    p = 2
    image = rng.normal(0, 1, (H, W, C))
    dim = p * p * C  # 12
    d_model = 4
    E = rng.normal(0, 1, (dim, d_model))
    N = (H // p) * (W // p)  # 16
    E_pos = rng.normal(0, 1, (N + 1, d_model))
    cls_token = rng.normal(0, 1, d_model)
    result = geron_vit_patch_embedding(image, p, E, E_pos, cls_token)
    assert isinstance(result, dict)
    assert "embeddings" in result
    assert "patches" in result
    assert "n_patches" in result
    assert "d_model" in result
    assert result["n_patches"] == N
    assert result["d_model"] == d_model
    embeddings = result["embeddings"]
    assert len(embeddings) == N + 1
    assert len(embeddings[0]) == d_model
    patches = result["patches"]
    assert len(patches) == N
    assert len(patches[0]) == dim


def test_grvit_edge():
    """Test edge cases."""
    # Smallest valid case from docstring: 2x2 image, patch_size 1
    img = [[1.0, 2.0], [3.0, 4.0]]
    p = 1
    E = [[1.0]]
    result = geron_vit_patch_embedding(img, p, E)
    assert isinstance(result, dict)
    assert result["n_patches"] == 4
    embeddings = result["embeddings"]
    # CLS slot (defaults to zeros) plus 4 patches, each width 1
    assert len(embeddings) == 5
    assert len(embeddings[0]) == 1
    # CLS slot is zeros when cls_token is omitted
    assert embeddings[0][0] == 0.0


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grvit as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
