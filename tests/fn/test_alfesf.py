"""Tests for alfesf.esmfold_lm_only (an alias of esmfold_confidence)."""

from morie.fn import _array_core as np
from morie.fn.alfesf import esmfold_lm_only


def test_alfesf_basic():
    """pLDDT is the expectation of the binned LDDT distribution: one value
    per residue, inside [0, 100]."""
    lddt_logits = np.random.default_rng(42).normal(0, 1, (30, 50))
    result = esmfold_lm_only(lddt_logits)
    assert isinstance(result, dict)
    assert "plddt" in result and len(result["plddt"]) == 30
    assert all(0.0 <= v <= 100.0 for v in result["plddt"])
    assert result["n_lddt_bins"] == 50


def test_alfesf_edge():
    """A single residue and a single bin still give one pLDDT value, and
    a one-dimensional input is refused as not being binned logits."""
    result = esmfold_lm_only(np.random.default_rng(42).normal(0, 1, (1, 3)))
    assert len(result["plddt"]) == 1
    try:
        esmfold_lm_only(np.random.default_rng(42).normal(0, 1, 100))
    except ValueError as e:
        assert "2-D" in str(e)
    else:
        raise AssertionError("1-D logits must be refused")
