"""contour is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.contour import contour


def test_contour_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        contour()
