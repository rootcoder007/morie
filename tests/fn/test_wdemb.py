"""wdemb: word-embedding lookup (Mikolov et al. 2013)."""

import numpy as np
import pytest

from morie.fn.wdemb import word_embedding as we


def test_wdemb_is_a_row_gather_from_the_table():
    E = np.arange(20.0).reshape(5, 4)
    ids = np.array([0, 3, 1])
    assert np.asarray(we(ids, E=E)["tensor"]) == pytest.approx(E[ids])


def test_wdemb_repeated_ids_give_identical_rows():
    """A lookup is deterministic: the same token always maps to the same
    vector within a call."""
    E = np.arange(12.0).reshape(3, 4)
    out = np.asarray(we(np.array([2, 2, 2]), E=E)["tensor"])
    assert np.allclose(out, out[0])


def test_wdemb_output_shape_is_ids_by_d_model():
    E = np.zeros((7, 5))
    r = we(np.array([1, 2, 3, 4]), E=E)
    assert np.asarray(r["tensor"]).shape == (4, 5)
    assert tuple(r["shape"]) == (4, 5)


def test_wdemb_builds_a_table_when_none_is_given():
    r = we(np.array([0, 1, 2]), vocab_size=10, d_model=6, seed=3)
    assert np.asarray(r["E"]).shape == (10, 6)
    assert np.asarray(r["tensor"]).shape == (3, 6)


def test_wdemb_generated_table_is_reproducible():
    a = np.asarray(we(np.array([0, 1]), vocab_size=8, d_model=4, seed=11)["E"])
    b = np.asarray(we(np.array([0, 1]), vocab_size=8, d_model=4, seed=11)["E"])
    assert a == pytest.approx(b, abs=0.0)


def test_wdemb_rejects_an_out_of_vocabulary_id():
    """Silently wrapping or clipping an OOV id yields a plausible vector for
    a token that does not exist -- the failure mode worth raising on."""
    E = np.zeros((4, 3))
    with pytest.raises((IndexError, ValueError)):
        we(np.array([9]), E=E)
