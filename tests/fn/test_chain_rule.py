"""Tests for morie.fn.chain_rule: values recomputed from first principles."""

from morie.fn.chain_rule import chain_rule


def test_both_factorizations():
    r = chain_rule(0.5, 0.4, 0.25, 0.8)
    assert r["via_a"] == 0.5 * 0.4
    assert r["via_b"] == 0.25 * 0.8
    assert r["p_and"] == r["via_a"]
