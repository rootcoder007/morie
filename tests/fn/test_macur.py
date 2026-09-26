"""macur is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.macur import macur


def test_macur_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        macur()
