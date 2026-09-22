"""Tests for alfcrp.alphafold_cropping."""

from morie.fn import _array_core as np

from morie.fn.alfcrp import alphafold_cropping


def test_alfcrp_basic():
    """Test basic functionality."""
    seqlen = 100
    crop_size = 100
    result = alphafold_cropping(seqlen, crop_size)
    assert isinstance(result, dict)
    assert result["estimate"] == crop_size


def test_alfcrp_edge():
    """Test edge cases."""
    seqlen = 100
    crop_size = 100
    result = alphafold_cropping(seqlen, crop_size)
    assert isinstance(result, dict)
    assert result["idx"] == list(range(0, seqlen))
    assert result["startmax"] == seqlen - crop_size + 1
    assert result["method"] == "AlphaFold contiguous residue cropping"
