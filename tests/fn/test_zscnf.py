"""zscnf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zscnf import contour_fill


def test_zscnf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        contour_fill(data=None)
