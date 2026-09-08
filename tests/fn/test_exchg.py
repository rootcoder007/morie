"""Tests for exchg.exchangeability_assumption."""

import pytest

from morie.fn.exchg import exchangeability_assumption

DAG = {"U": ["T", "Y"], "T": ["Y"]}


def test_exchg_basic():
    assert exchangeability_assumption(DAG, "T", "Y")["holds"] is False
    out = exchangeability_assumption(DAG, "T", "Y", X=("U",))
    assert out["holds"] is True
    assert out["adjustment_set"] == ("U",)


def test_exchg_edge():
    # conditioning on a descendant of T violates the criterion
    dag = {"U": ["T", "Y"], "T": ["Y", "M"]}
    assert exchangeability_assumption(dag, "T", "Y", X=("U", "M"))["holds"] is False
