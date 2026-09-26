"""nbdnl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbdnl import nbdnl


def test_nbdnl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbdnl()
