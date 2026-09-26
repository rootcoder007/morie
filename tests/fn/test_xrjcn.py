"""xrjcn re-exports the real join_count from jjmsta."""

from morie.fn.jjmsta import join_count as canonical
from morie.fn.xrjcn import join_count


def test_xrjcn_is_the_canonical_implementation():
    assert join_count is canonical
