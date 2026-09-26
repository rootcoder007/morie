"""trdns is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trdns import trdns


def test_trdns_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trdns()
