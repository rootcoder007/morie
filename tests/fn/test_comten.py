"""comten is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.comten import community_modularity


def test_comten_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        community_modularity(G=None, partition=None)
