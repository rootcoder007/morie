"""tsstl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tsstl import tsstl


def test_tsstl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tsstl()
