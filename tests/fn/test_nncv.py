"""nncv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nncv import nncv


def test_nncv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nncv()
