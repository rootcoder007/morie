"""wqcr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqcr import wqcr


def test_wqcr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqcr()
