"""Tests for avalon.avalon_fingerprint."""

from morie.fn import _array_core as np

from morie.fn.avalon import avalon_fingerprint


def test_avalon_basic():
    """Test basic functionality against the documented formula."""
    smiles = "c1ccccc1"
    n_bits = 128
    result = avalon_fingerprint(smiles, n_bits=n_bits, maxpath=3)

    # The function returns a RichResult with payload dict semantics
    assert hasattr(result, "payload")
    payload = result.payload if hasattr(result, "payload") else result
    assert isinstance(payload, dict)

    # Documented return keys
    assert "bits" in payload
    assert "on" in payload
    assert "features" in payload
    assert "n_features" in payload
    assert "n_on" in payload
    assert "n_collisions" in payload
    assert "density" in payload
    assert "n_atoms" in payload
    assert "n_bonds" in payload
    assert "n_rings" in payload
    assert "n_hydrogens" in payload
    assert "n_bits" in payload
    assert "maxpath" in payload
    assert "classes" in payload
    assert "method" in payload

    # Shape and value invariants for benzene with these settings
    assert payload["n_bits"] == n_bits
    assert isinstance(payload["bits"], list)
    assert len(payload["bits"]) == n_bits
    assert all(bit in (0, 1) for bit in payload["bits"])
    assert payload["n_on"] == sum(payload["bits"])
    assert payload["density"] == payload["n_on"] / float(n_bits)
    assert payload["n_features"] == len(payload["features"])
    assert payload["n_collisions"] >= 0
    # benzene: 6 atoms, 6 aromatic C-C bonds, no small rings beyond the
    # 6-membered ring, so exactly one ring
    assert payload["n_atoms"] == 6
    assert payload["n_rings"] == 1
    assert payload["maxpath"] == 3


def test_avalon_edge():
    """Test edge cases on argument shapes and documented constraints."""
    # smiles must be a string per the docstring
    smiles = "CCO"  # ethanol
    n_bits = 64
    result = avalon_fingerprint(smiles, n_bits=n_bits, maxpath=5)

    payload = result.payload if hasattr(result, "payload") else result
    assert isinstance(payload, dict)
    assert payload["n_bits"] == n_bits
    assert len(payload["bits"]) == n_bits

    # explicit classes restriction: only path features
    result2 = avalon_fingerprint(
        "c1ccccc1", n_bits=256, maxpath=4, classes=["path"]
    )
    payload2 = result2.payload if hasattr(result2, "payload") else result2
    assert payload2["classes"] == ["path"]
    assert payload2["n_bits"] == 256

    # explicit classes restriction: only bond features
    result3 = avalon_fingerprint(
        "c1ccccc1", n_bits=256, maxpath=4, classes=["bond"]
    )
    payload3 = result3.payload if hasattr(result3, "payload") else result3
    assert payload3["classes"] == ["bond"]

    # When classes is None, the function returns the full class list
    # (length equals the number of supported feature classes).
    result4 = avalon_fingerprint("c1ccccc1", n_bits=128, maxpath=3)
    payload4 = result4.payload if hasattr(result4, "payload") else result4
    assert payload4["classes"] is not None
    assert isinstance(payload4["classes"], list)
    assert len(payload4["classes"]) >= 1
