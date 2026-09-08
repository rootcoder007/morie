"""Tests for alcnp.alammar_chain_prompting."""

from morie.fn.alcnp import alammar_chain_prompting


def test_alcnp_basic():
    out = alammar_chain_prompting("3",
        [lambda y, x: x, lambda y, x: y + "!"], lambda p: p)
    assert out["final_output"] == "3!"


def test_alcnp_edge():
    import pytest
    with pytest.raises(ValueError, match="no prompts"):
        alammar_chain_prompting("x", [], lambda p: p)
