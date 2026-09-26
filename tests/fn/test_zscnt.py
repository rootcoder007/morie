"""zscnt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zscnt import contour_lines


def test_zscnt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        contour_lines(data=None)
