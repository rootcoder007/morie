"""ftest is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ftest import ftest


def test_ftest_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ftest()
