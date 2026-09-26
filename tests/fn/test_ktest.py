"""ktest is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ktest import ktest


def test_ktest_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ktest()
