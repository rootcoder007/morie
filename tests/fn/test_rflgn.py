"""rflgn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rflgn import rflgn


def test_rflgn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rflgn()
