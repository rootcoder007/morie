"""xrwis is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrwis import w_islands


def test_xrwis_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        w_islands(data=None)
