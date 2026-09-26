"""gacurv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gacurv import gacurv


def test_gacurv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gacurv()
