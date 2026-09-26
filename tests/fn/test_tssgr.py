"""tssgr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tssgr import tssgr


def test_tssgr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tssgr()
