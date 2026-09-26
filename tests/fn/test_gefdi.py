"""gefdi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gefdi import gefdi


def test_gefdi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gefdi()
