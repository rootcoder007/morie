"""cstrr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cstrr import cstrr


def test_cstrr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cstrr()
