"""dlaumn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dlaumn import dlaumn


def test_dlaumn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dlaumn()
