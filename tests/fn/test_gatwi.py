"""gatwi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gatwi import gatwi


def test_gatwi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gatwi()
