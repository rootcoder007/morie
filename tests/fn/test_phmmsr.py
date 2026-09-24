"""Tests for phmmsr.profile_hmm_search."""

from morie.fn import _array_core as np

from morie.fn.phmmsr import profile_hmm_search, cheatsheet


def test_phmmsr_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    profile = cheatsheet()
    db = [list(rng.integers(0, 4, 30)) for _ in range(10)]
    
    result = profile_hmm_search(profile, db)
    assert isinstance(result, dict)


def test_phmmsr_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    profile = cheatsheet()
    db = [list(rng.integers(0, 4, 15)) for _ in range(5)]
    
    result = profile_hmm_search(profile, db)
    assert isinstance(result, dict)
