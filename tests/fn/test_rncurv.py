"""rncurv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rncurv import rncurv


def test_rncurv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rncurv()
