"""winklt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.winklt import winklt


def test_winklt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        winklt()
