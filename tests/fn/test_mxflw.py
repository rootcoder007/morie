"""Tests for mxflw (max flow)."""

from morie.fn import _array_core as np

from morie.fn.mxflw import max_flow


def test_max_flow_basic():
    cap = np.array(
        [
            [0, 10, 10, 0],
            [0, 0, 0, 10],
            [0, 0, 0, 10],
            [0, 0, 0, 0],
        ],
        dtype=float,
    )
    r = max_flow(cap, source=0, sink=3)
    assert r.value == 20


def test_cheatsheet():
    from morie.fn.mxflw import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
