"""maco2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.maco2 import maco2


def test_maco2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        maco2()
