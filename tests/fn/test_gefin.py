"""gefin is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gefin import gefin


def test_gefin_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gefin()
