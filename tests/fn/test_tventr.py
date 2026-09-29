"""Tests for tventr (total variation distance)."""

import pytest

from morie.fn.tventr import tventr


def test_total_variation_recomputed():
    p = [2.0, 3.0, 5.0]
    q = [1.0, 1.0, 2.0]
    P = [v / 10 for v in p]
    Q = [v / 4 for v in q]
    r = tventr(p, q)
    assert r["value"] == pytest.approx(0.5 * sum(abs(a - b) for a, b in zip(P, Q)), rel=1e-14)
    assert tventr(p, p)["value"] == 0.0
