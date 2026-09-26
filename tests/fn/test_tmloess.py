"""tmloess is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tmloess import tmloess


def test_tmloess_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tmloess()
