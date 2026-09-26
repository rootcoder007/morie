"""mgwrbnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mgwrbnd import mgwrbnd


def test_mgwrbnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mgwrbnd(bw=None, se_bw=None)
