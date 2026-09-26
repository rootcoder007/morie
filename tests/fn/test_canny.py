"""canny is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.canny import canny


def test_canny_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        canny()
