"""tsspn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tsspn import tsspn


def test_tsspn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tsspn()
