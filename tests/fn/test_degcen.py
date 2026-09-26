"""degcen is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.degcen import degree_centrality


def test_degcen_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        degree_centrality(G=None)
