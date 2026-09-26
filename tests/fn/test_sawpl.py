"""sawpl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sawpl import sawpl


def test_sawpl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sawpl()
