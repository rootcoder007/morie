"""Tests for alfmpv.alphafold_multimer."""

import pytest

from morie.fn import _array_core as np

from morie.fn.alfmpv import alphafold_multimer


def test_alfmpv_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # per-chain MSAs are dicts that carry at least a 'species' field
    msas = [
        {"species": "human", "msa": rng.normal(0, 1, (10, 20))},
        {"species": "mouse", "msa": rng.normal(0, 1, (10, 20))},
        {"species": "human", "msa": rng.normal(0, 1, (10, 20))},
    ]
    result = alphafold_multimer(msas=msas)
    assert isinstance(result, dict)


def test_alfmpv_edge():
    """Test edge cases: omitting both chains and msas must raise."""
    with pytest.raises(ValueError):
        alphafold_multimer()
