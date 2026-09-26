"""hsdwn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hsdwn import hsdwn


def test_hsdwn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hsdwn()
