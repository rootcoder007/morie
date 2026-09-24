"""Tests for kmmae.kamath_multimodal_mae."""

import math

from morie.fn import _array_core as np

from morie.fn.kmmae import kamath_multimodal_mae


def test_kmmae_basic():
    """Test basic functionality with a single modality."""
    rng = np.random.default_rng(42)
    n = 100
    visible = rng.normal(0, 1, n).tolist()
    mask_arr = rng.integers(0, 2, n).tolist()
    n_masked = int(np.sum(np.asarray(mask_arr).astype(bool)))
    gold = rng.normal(0, 1, n_masked).tolist()

    x_visible = {"img": visible}
    x_masked_true = {"img": gold}
    masks = {"img": mask_arr}
    decoders = {"img": lambda v, m: [0.0] * int(np.sum(np.asarray(m).astype(bool)))}

    result = kamath_multimodal_mae(x_visible, x_masked_true, masks, decoders=decoders)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "loss" in result
    assert "per_modality" in result
    assert "modalities" in result
    assert "n_masked" in result
    assert "n" in result
    assert "method" in result
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["loss"])
    assert result["n_masked"] == n_masked
    assert result["n"] == 1
    assert result["modalities"] == ["img"]
    assert "img" in result["per_modality"]


def test_kmmae_edge():
    """Test edge case: single modality matching the docstring example."""
    x_visible = {"img": [1.0, 1.0]}
    x_masked_true = {"img": [2.0, 4.0]}
    masks = {"img": [False, False, True, True]}
    decoders = {"img": lambda v, m: [2.0, 3.0]}

    result = kamath_multimodal_mae(x_visible, x_masked_true, masks, decoders=decoders)
    assert isinstance(result, dict)
    assert result["estimate"] == 1.0
    assert result["loss"] == 1.0
    assert result["n_masked"] == 2
    assert result["n"] == 1
    assert result["per_modality"] == {"img": 1.0}
    assert result["modalities"] == ["img"]
    assert result["method"] == "Multimodal MAE squared reconstruction loss"


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmmae as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
