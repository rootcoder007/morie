"""opass is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opass import opass


def test_opass_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opass()
