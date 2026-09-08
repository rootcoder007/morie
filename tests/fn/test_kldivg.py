"""Tests for kldivg.kldivg."""

from morie.fn import _array_core as np
import pytest

from morie.fn.kldivg import kldivg


def test_kldivg_is_zero_only_for_identical_distributions():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    assert float(np.asarray(kldivg(p, p)["value"], dtype=float)) == pytest.approx(0.0, abs=1e-12)


def test_kldivg_matches_the_hand_computed_value_in_bits():
    """A two-point case small enough to write out: p = (3/4, 1/4) against a
    fair coin gives 1 - H(p) = 1 - 0.811278 bits."""
    d = float(np.asarray(kldivg([0.75, 0.25], [0.5, 0.5], base=2.0)["value"], dtype=float))
    want = 0.75 * np.log2(0.75 / 0.5) + 0.25 * np.log2(0.25 / 0.5)
    assert d == pytest.approx(want, rel=1e-9)
    assert d == pytest.approx(1.0 - 0.8112781244591328, rel=1e-9)


def test_kldivg_is_asymmetric():
    """The asymmetry is the whole reason KL is a divergence and not a metric."""
    p, q = [0.9, 0.1], [0.5, 0.5]
    fwd = float(np.asarray(kldivg(p, q)["value"], dtype=float))
    rev = float(np.asarray(kldivg(q, p)["value"], dtype=float))
    assert fwd > 0 and rev > 0
    assert abs(fwd - rev) > 1e-3


def test_kldivg_base_changes_only_the_units():
    p, q = [0.7, 0.3], [0.4, 0.6]
    bits = float(np.asarray(kldivg(p, q, base=2.0)["value"], dtype=float))
    nats = float(np.asarray(kldivg(p, q, base=np.e)["value"], dtype=float))
    assert bits == pytest.approx(nats / np.log(2), rel=1e-9)


def test_kldivg_rejects_mismatched_shapes_and_negative_mass():
    with pytest.raises(ValueError, match="shape mismatch"):
        kldivg([0.5, 0.5], [0.3, 0.3, 0.4])
    with pytest.raises(ValueError, match="non-negative"):
        kldivg([-0.1, 1.1], [0.5, 0.5])
