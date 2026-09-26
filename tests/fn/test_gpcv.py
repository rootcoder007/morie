"""gpcv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gpcv import gpcv


def test_gpcv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gpcv()
