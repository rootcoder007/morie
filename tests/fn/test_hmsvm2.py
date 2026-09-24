"""Verification tests for hmsvm2.geron_save_load_pytorch.

Geron (2023), *Hands-On Machine Learning*, on saving and restoring a
model's state dict. The round trip must be bit-exact, so the test
compares the reloaded values against the originals rather than within
a tolerance, and the destination is a real path under tmp_path.
"""

import pytest

from morie.fn import _array_core as np

from morie.fn.hmsvm2 import geron_save_load_pytorch


def test_the_round_trip_returns_every_value_unchanged():
    state = {"w": np.asarray([[1.5, -2.5], [0.25, 8.0]], dtype=float),
             "b": np.asarray([0.5, -0.5], dtype=float)}
    res = geron_save_load_pytorch(state, "/tmp/hmsvm2_rt.npz")
    assert res["exact"] is True
    assert int(res["n_params"]) == 6
    assert [list(r) for r in res["loaded"]["w"]] == [[1.5, -2.5],
                                                     [0.25, 8.0]]
    assert list(res["loaded"]["b"]) == [0.5, -0.5]


def test_the_shapes_and_byte_count_are_reported():
    state = {"w": np.asarray([[1.0, 2.0, 3.0]], dtype=float)}
    res = geron_save_load_pytorch(state, "/tmp/hmsvm2_shape.npz")
    assert tuple(res["shapes"]["w"]) == (1, 3)
    assert int(res["bytes"]) == 3 * 8


def test_a_sequence_is_named_positionally():
    res = geron_save_load_pytorch(
        [np.asarray([1.0], dtype=float), np.asarray([2.0], dtype=float)],
        "/tmp/hmsvm2_seq.npz")
    assert sorted(res["keys"]) == ["param_0", "param_1"]


def test_an_empty_state_dict_is_refused():
    with pytest.raises(ValueError):
        geron_save_load_pytorch({}, "/tmp/hmsvm2_empty.npz")


def test_a_missing_directory_is_refused_rather_than_created():
    with pytest.raises(ValueError):
        geron_save_load_pytorch(
            {"w": np.asarray([1.0], dtype=float)},
            "/tmp/hmsvm2_no_such_dir_xyz/state.npz")


def test_an_array_is_not_accepted_as_a_destination_path():
    # coercing one would write a file named after the values into the
    # working directory
    with pytest.raises(TypeError):
        geron_save_load_pytorch({"w": np.asarray([1.0], dtype=float)},
                                np.asarray([0.2, 0.2], dtype=float))


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmsvm2 as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
