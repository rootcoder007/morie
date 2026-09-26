"""wqcnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqcnd import wqcnd


def test_wqcnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqcnd()
