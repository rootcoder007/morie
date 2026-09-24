"""Tests for kmp2.kamath_p_tuning_v2."""

import pytest

from morie.fn import _array_core as np

from morie.fn.kmp2 import kamath_p_tuning_v2


def test_kmp2_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # Two layers, each with prefix shape (2, 3) and input shape (4, 3)
    pre_K_l0 = rng.normal(0, 1, (2, 3)).tolist()
    pre_V_l0 = rng.normal(0, 1, (2, 3)).tolist()
    pre_K_l1 = rng.normal(0, 1, (2, 3)).tolist()
    pre_V_l1 = rng.normal(0, 1, (2, 3)).tolist()
    inp_K_l0 = rng.normal(0, 1, (4, 3)).tolist()
    inp_V_l0 = rng.normal(0, 1, (4, 3)).tolist()
    inp_K_l1 = rng.normal(0, 1, (4, 3)).tolist()
    inp_V_l1 = rng.normal(0, 1, (4, 3)).tolist()

    prefixes_by_layer = [
        (pre_K_l0, pre_V_l0),
        (pre_K_l1, pre_V_l1),
    ]
    inputs_by_layer = [
        (inp_K_l0, inp_V_l0),
        (inp_K_l1, inp_V_l1),
    ]

    result = kamath_p_tuning_v2(prefixes_by_layer, inputs_by_layer)
    assert isinstance(result, dict)
    assert "K" in result and "V" in result
    assert "prefix_len" in result and "n_layers" in result
    assert "n_trainable" in result
    assert result["n_layers"] == 2
    assert result["prefix_len"] == [2, 2]
    # 2 layers * (2*3 + 2*3) = 24 trainable parameters
    assert result["n_trainable"] == 24
    assert len(result["K"]) == 2
    assert len(result["V"]) == 2
    # After stacking prefix (2, 3) with input (4, 3), each K/V is (6, 3)
    for k in result["K"]:
        assert len(k) == 6
        assert len(k[0]) == 3
    for v in result["V"]:
        assert len(v) == 6
        assert len(v[0]) == 3


def test_kmp2_edge():
    """Test edge cases: mismatched prefix/input layer counts raises."""
    pre = [([[1.0, 1.0]], [[2.0, 2.0]])]
    inp = [
        ([[0.0, 0.0], [3.0, 3.0]], [[0.0, 0.0], [4.0, 4.0]]),
        ([[0.0, 0.0], [3.0, 3.0]], [[0.0, 0.0], [4.0, 4.0]]),
    ]
    with pytest.raises(ValueError):
        kamath_p_tuning_v2(pre, inp)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmp2 as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
