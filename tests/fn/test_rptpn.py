"""Tests for rptpn.repetition_penalty."""

from morie.fn import _array_core as np
import pytest

from morie.fn.rptpn import repetition_penalty


def test_rptpn_penalises_both_signs_toward_less_likely():
    """Keskar et al. (2019, CTRL): positive logits are divided, negative
    multiplied -- either way the penalised token loses probability."""
    z = np.array([2.0, -2.0, 1.0])
    r = repetition_penalty(z, generated=[0, 1], alpha=2.0)
    out = np.asarray(r["tensor"], dtype=float)
    assert out[0] == pytest.approx(1.0, abs=1e-12)    # 2 / 2
    assert out[1] == pytest.approx(-4.0, abs=1e-12)   # -2 * 2
    assert out[2] == pytest.approx(1.0, abs=1e-12)    # untouched
    # Softmax probability of both penalised tokens strictly drops.
    p_before = np.exp(z) / np.exp(z).sum()
    p_after = np.exp(out) / np.exp(out).sum()
    assert p_after[0] < p_before[0] and p_after[1] < p_before[1]


def test_rptpn_alpha_one_is_a_no_op():
    z = np.array([0.5, -0.5])
    out = np.asarray(repetition_penalty(z, generated=[0, 1], alpha=1.0)["tensor"], dtype=float)
    np.testing.assert_array_equal(out, z)


def test_rptpn_out_of_range_ids_are_ignored():
    z = np.array([1.0, 2.0])
    r = repetition_penalty(z, generated=[5, -3, 1], alpha=2.0)
    assert np.asarray(r["penalised_idx"]).tolist() == [1]
