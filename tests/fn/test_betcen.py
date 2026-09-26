"""betcen is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.betcen import betweenness_centrality


def test_betcen_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        betweenness_centrality(G=None)
