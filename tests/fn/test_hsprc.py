"""hsprc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hsprc import hsprc


def test_hsprc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hsprc()
