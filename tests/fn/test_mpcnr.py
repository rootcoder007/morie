"""mpcnr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mpcnr import mpcnr


def test_mpcnr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mpcnr()
