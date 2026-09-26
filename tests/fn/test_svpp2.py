"""svpp2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svpp2 import party_pos_2d


def test_svpp2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        party_pos_2d(data=None)
