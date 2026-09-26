"""rfsmt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rfsmt import rfsmt


def test_rfsmt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rfsmt()
