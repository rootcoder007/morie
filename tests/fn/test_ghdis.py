"""ghdis is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghdis import ghdis


def test_ghdis_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghdis()
