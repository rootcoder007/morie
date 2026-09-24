"""Tests for hmclc.geron_classification_localization."""

import math

from morie.fn import _array_core as np

from morie.fn.hmclc import geron_classification_localization


def test_hmclc_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    image = rng.normal(0, 1, (10, 10))
    # Model returns K=2 class scores + 4 box values = 6 total
    model = lambda img: np.array([0.0, 0.0, 5.0, 5.0, 2.0, 4.0])
    result = geron_classification_localization(image, model)
    assert isinstance(result, dict)
    # All keys named in the return statement must be present
    expected_keys = {"class_probs", "predicted_class", "box", "box_corners",
                     "iou", "loss", "loss_class", "loss_box", "n_classes",
                     "estimate", "n", "method"}
    assert expected_keys.issubset(set(result.keys()))
    # softmax([0, 0]) == [0.5, 0.5] -- derivable
    assert result["class_probs"][0] == [0.5, 0.5]
    # box read off directly (centre-size)
    assert result["box"][0] == [5.0, 5.0, 2.0, 4.0]
    # corners = [x - w/2, y - h/2, x + w/2, y + h/2]
    assert result["box_corners"][0] == [4.0, 3.0, 6.0, 7.0]
    # n_classes inferred as len(output) - 4 == 2
    assert result["n_classes"] == 2


def test_hmclc_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    image = rng.normal(0, 1, (10, 10))
    model = lambda img: np.array([0.0, 0.0, 5.0, 5.0, 2.0, 4.0])
    # Perfect box match: IoU == 1, loss_box == 0, loss == log(2)
    result = geron_classification_localization(
        image, model,
        gt_class=[0],
        gt_box=[[5.0, 5.0, 2.0, 4.0]],
    )
    assert isinstance(result, dict)
    assert result["iou"][0] == 1.0
    assert result["loss_box"] == 0.0
    # uniform class probabilities -> cross-entropy = log(2)
    assert math.isclose(result["loss"], math.log(2), rel_tol=1e-9)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmclc as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
