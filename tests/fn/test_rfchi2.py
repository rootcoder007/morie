"""rfchi2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rfchi2 import rfchi2


def test_rfchi2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rfchi2()
