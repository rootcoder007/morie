"""Tests for hmvit.geron_vision_transformer."""

import doctest as _doctest

from morie.fn import _array_core as np
from morie.fn.hmvit import geron_vision_transformer

import morie.fn.hmvit as _doctest_module


def test_hmvit_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, (8, 8))
    result = geron_vision_transformer(
        image, patch_size=2, n_layers=1, d_model=8, n_heads=2, n_classes=3
    )
    assert isinstance(result, dict)
    # Keys named in the return statement
    for key in ("logits", "cls", "tokens", "n_patches", "seq_len",
                "total_params", "estimate", "n", "method"):
        assert key in result
    # 8x8 image with 2x2 patches -> 4x4 = 16 patches, seq_len = 17
    assert int(result["n_patches"]) == 16
    assert int(result["seq_len"]) == 17
    assert len(result["logits"]) == 3


def test_hmvit_edge():
    """Test edge cases."""
    # Smallest valid (H, W) configuration: 4x4 image, 2x2 patches
    image = np.random.default_rng(42).normal(0, 1, (4, 4))
    result = geron_vision_transformer(
        image, patch_size=2, n_layers=1, d_model=4, n_heads=2, n_classes=2
    )
    assert isinstance(result, dict)
    assert "logits" in result
    assert "n_patches" in result
    assert "seq_len" in result
    # 4x4 image with 2x2 patches -> 4 patches, seq_len = 5
    assert int(result["n_patches"]) == 4
    assert int(result["seq_len"]) == 5
    assert len(result["logits"]) == 2


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
