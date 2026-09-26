"""enlnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.enlnd import enlnd


def test_enlnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enlnd()
