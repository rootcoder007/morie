"""ghdst is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghdst import ghdst


def test_ghdst_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghdst()
