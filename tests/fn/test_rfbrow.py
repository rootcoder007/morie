"""rfbrow is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rfbrow import rfbrow


def test_rfbrow_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rfbrow()
