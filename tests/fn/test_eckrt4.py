"""eckrt4 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.eckrt4 import eckrt4


def test_eckrt4_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        eckrt4()
