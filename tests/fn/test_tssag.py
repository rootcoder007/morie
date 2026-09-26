"""tssag is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tssag import tssag


def test_tssag_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tssag()
