"""tmses is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tmses import tmses


def test_tmses_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tmses()
