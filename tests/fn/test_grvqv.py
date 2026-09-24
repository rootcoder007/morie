"""Tests for grvqv.geron_vq_vae_codebook_loss."""

from morie.fn import _array_core as np

from morie.fn.grvqv import geron_vq_vae_codebook_loss


def test_grvqv_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    z_e = np.random.default_rng(42).normal(0, 1, 100)
    z_q = np.random.default_rng(42).normal(0, 1, 100)
    codebook = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = geron_vq_vae_codebook_loss(x, z_e, z_q, codebook, beta)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grvqv_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    z_e = np.random.default_rng(42).normal(0, 1, 100)
    z_q = np.random.default_rng(42).normal(0, 1, 100)
    codebook = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = geron_vq_vae_codebook_loss(x, z_e, z_q, codebook, beta)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grvqv as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
