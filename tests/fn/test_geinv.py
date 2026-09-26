"""geinv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.geinv import geinv


def test_geinv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        geinv()
