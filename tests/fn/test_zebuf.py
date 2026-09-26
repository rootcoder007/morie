"""zebuf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zebuf import buffer_exposure


def test_zebuf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        buffer_exposure(data=None)
