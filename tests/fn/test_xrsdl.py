"""xrsdl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrsdl import sdem_ml


def test_xrsdl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sdem_ml(data=None)
