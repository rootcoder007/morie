"""tsgwt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tsgwt import tsgwt


def test_tsgwt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tsgwt()
