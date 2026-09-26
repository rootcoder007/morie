"""enpet is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.enpet import enpet


def test_enpet_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enpet()
