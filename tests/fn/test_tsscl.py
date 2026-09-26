"""tsscl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tsscl import tsscl


def test_tsscl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tsscl()
