"""wqeut is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqeut import wqeut


def test_wqeut_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqeut()
