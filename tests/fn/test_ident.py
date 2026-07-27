"""Tests for ident.identifiability_conditions."""

import numpy as np
import pytest

from morie.fn.ident import identifiability_conditions

DAG = {"U": ["T", "Y"], "T": ["Y"]}


def test_ident_basic():
    T = np.array([1, 0, 1, 0])
    S = np.array(["a", "a", "b", "b"])
    out = identifiability_conditions(DAG, "T", "Y", Z=("U",), treatment=T, strata=S)
    assert out["identifiable"] is True
    assert out["positivity"] is True


def test_ident_edge():
    with pytest.raises(ValueError):
        identifiability_conditions(DAG, "T", "Y", Z=("U",), treatment=[1, 0])  # no strata
    out = identifiability_conditions(DAG, "T", "Y")  # no data: positivity unknown
    assert out["positivity"] is None and out["identifiable"] is None
