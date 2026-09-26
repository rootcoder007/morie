"""rfgss is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rfgss import rfgss


def test_rfgss_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rfgss()
