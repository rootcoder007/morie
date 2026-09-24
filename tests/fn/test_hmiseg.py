"""Tests for hmiseg.geron_image_segmentation."""

import math

from morie.fn import _array_core as np

from morie.fn.hmiseg import geron_image_segmentation


def test_hmiseg_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    image = rng.uniform(0.0, 1.0, (4, 4, 3))
    n_clusters = 2
    seed = 42
    result = geron_image_segmentation(image, n_clusters, seed)
    assert isinstance(result, dict)
    for key in ("segmented", "labels", "palette", "inertia",
                "compression_ratio", "estimate", "n", "method"):
        assert key in result
    assert result["segmented"].shape == (4, 4, 3)
    assert result["labels"].shape == (4, 4)
    assert result["palette"].shape == (2, 3)
    assert result["n"] == 16
    assert math.isfinite(float(result["inertia"]))
    assert float(result["compression_ratio"]) >= 0.0


def test_hmiseg_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    image = rng.uniform(0.0, 1.0, (2, 3, 1))
    n_clusters = 1
    seed = 0
    result = geron_image_segmentation(image, n_clusters, seed)
    assert isinstance(result, dict)
    for key in ("segmented", "labels", "palette", "inertia",
                "compression_ratio", "estimate", "n", "method"):
        assert key in result
    assert result["segmented"].shape == (2, 3, 1)
    assert result["palette"].shape == (1, 1)
    assert result["labels"].shape == (2, 3)
    assert result["n"] == 6
    assert math.isfinite(float(result["inertia"]))


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmiseg as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
