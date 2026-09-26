"""erosn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.erosn import erosn


def test_erosn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        erosn()
