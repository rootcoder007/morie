"""Tests for alvit.alammar_vit_patch_embedding."""

from morie.fn.alvit import alammar_vit_patch_embedding


def test_alvit_basic():
    from morie.fn import _array_core as np
    out = alammar_vit_patch_embedding(np.arange(16.0).reshape(4, 4), 2,
                                      np.eye(4))
    assert out["n_patches"] == 4


def test_alvit_edge():
    import pytest
    from morie.fn import _array_core as np
    with pytest.raises(ValueError, match="tile"):
        alammar_vit_patch_embedding(np.zeros((5, 4)), 2, np.eye(4))
