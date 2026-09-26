"""rbflnr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rbflnr import rbflnr


def test_rbflnr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rbflnr()
