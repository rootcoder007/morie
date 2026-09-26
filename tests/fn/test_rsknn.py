"""rsknn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsknn import rsknn


def test_rsknn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsknn()
