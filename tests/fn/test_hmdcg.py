"""Tests for hmdcg.geron_dcgan."""

import pytest
from morie.fn import _array_core as np

from morie.fn.hmdcg import geron_dcgan


def test_hmdcg_basic():
    """Test basic functionality."""
    # A 16x16 image with seed_shape=(4,4) and stride=2 needs exactly 2 layers.
    X = np.zeros((3, 16, 16))
    z_dim = 8
    filters = 4
    epochs = 50
    lr = 0.0002
    result = geron_dcgan(X, z_dim=z_dim, filters=filters, epochs=epochs, lr=lr)
    assert isinstance(result, dict)
    assert "generator_layers" in result
    assert "discriminator_layers" in result
    assert "n_layers" in result
    assert "sample_shape" in result
    assert result["sample_shape"] == (16, 16)
    assert result["n_layers"] == 2
    deconv_outs = [l["out"] for l in result["generator_layers"] if l["kind"] == "deconv"]
    assert deconv_outs == [8, 16]
    conv_outs = [l["out"] for l in result["discriminator_layers"] if l["kind"] == "conv"]
    assert conv_outs == [8, 4]
    assert result["generator_layers"][0]["params"] == 1152


def test_hmdcg_edge():
    """Test edge cases."""
    # 20 does not decompose as 4 * 2^L for any integer L, so this must raise.
    X = np.zeros((1, 20, 20))
    z_dim = 8
    filters = 4
    with pytest.raises(ValueError):
        geron_dcgan(X, z_dim=z_dim, filters=filters)
