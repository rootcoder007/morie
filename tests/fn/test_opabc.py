"""opabc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opabc import opabc


def test_opabc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opabc()
