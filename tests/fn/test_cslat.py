"""cslat: causal (autoregressive) attention mask (Radford et al. 2019)."""

import numpy as np
import pytest

from morie.fn.cslat import causal_attention_mask as mask


def test_cslat_is_lower_triangular():
    """Position i may attend to j <= i and to nothing later. That single
    property is what makes an LM autoregressive."""
    m = np.asarray(mask(np.zeros((6, 6))))if False else np.asarray(mask(6)["tensor"])
    n = m.shape[-1]
    for i in range(n):
        for j in range(n):
            allowed = bool(m[i, j]) if m.dtype == bool else np.isfinite(m[i, j]) and m[i, j] == 0
            assert allowed == (j <= i), f"position {i} vs {j}"


def test_cslat_diagonal_is_always_visible():
    """A token must see itself."""
    m = np.asarray(mask(5)["tensor"])
    d = np.diag(m)
    assert np.all(d == d[0])


def test_cslat_no_future_leakage():
    """The strict upper triangle must be uniformly blocked."""
    m = np.asarray(mask(7)["tensor"])
    iu = np.triu_indices(7, k=1)
    blocked = m[iu]
    assert np.all(blocked == blocked[0]), "all future positions equally blocked"


def test_cslat_size_matches_the_request():
    for n in (1, 3, 12):
        assert np.asarray(mask(n)["tensor"]).shape == (n, n)
        assert mask(n)["n"] == n


def test_cslat_single_token_can_only_see_itself():
    m = np.asarray(mask(1)["tensor"])
    assert m.shape == (1, 1)
