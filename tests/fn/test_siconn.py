"""siconn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.siconn import siconn


def test_siconn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        siconn()
