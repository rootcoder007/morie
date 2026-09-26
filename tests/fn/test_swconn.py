"""swconn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.swconn import swconn


def test_swconn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        swconn(W=None)
