"""rforns is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rforns import rforns


def test_rforns_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rforns()
