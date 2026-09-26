"""prjinv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.prjinv import prjinv


def test_prjinv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        prjinv()
