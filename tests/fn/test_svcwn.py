"""svcwn re-exports the real condorcet_winner from cndrc."""

from morie.fn.cndrc import condorcet_winner as canonical
from morie.fn.svcwn import condorcet_winner


def test_svcwn_is_the_canonical_implementation():
    assert condorcet_winner is canonical
