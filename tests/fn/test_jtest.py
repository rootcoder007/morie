"""jtest is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.jtest import jtest


def test_jtest_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        jtest()
