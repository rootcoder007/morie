"""Tests for morie.fn.swblk."""

from morie.fn.swblk import swblk

G = ["a", "b", "a", "c", "b", "a"]


def test_block_and_row_standardised():
    r = swblk(G)
    for i in range(6):
        for j in range(6):
            assert r.extra["W"][i][j] == (1.0 if i != j and G[i] == G[j] else 0.0)
    w = swblk(G, style="W").extra["W"]
    assert w[0] == [0.0, 0.0, 0.5, 0.0, 0.0, 0.5]
    assert w[3] == [0.0] * 6
