"""stvario is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stvario import stvario


def test_stvario_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stvario()
