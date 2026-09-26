"""sawvr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sawvr import sawvr


def test_sawvr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sawvr()
