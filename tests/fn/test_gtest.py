"""gtest is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gtest import gtest


def test_gtest_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gtest()
