"""datumx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.datumx import datumx


def test_datumx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        datumx()
