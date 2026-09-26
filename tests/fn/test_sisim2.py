"""sisim2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sisim2 import sisim2


def test_sisim2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sisim2()
