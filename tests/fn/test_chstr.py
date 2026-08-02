"""Tests for chstr.chain_structure."""

from morie.fn import _array_core as np
import pytest

from morie.fn.chstr import chain_structure


def test_chstr_basic():
    rng = np.random.default_rng(42)
    a = rng.normal(size=3000)
    b = a + rng.normal(scale=0.7, size=3000)
    c = b + rng.normal(scale=0.7, size=3000)
    out = chain_structure(a, b, c)
    assert out["consistent_with_chain"] is True


def test_chstr_edge():
    # collider data: a, c independent marginally -> not a chain signature
    rng = np.random.default_rng(0)
    a = rng.normal(size=3000)
    c = rng.normal(size=3000)
    b = a + c + rng.normal(scale=0.5, size=3000)
    assert chain_structure(a, b, c)["consistent_with_chain"] is False
