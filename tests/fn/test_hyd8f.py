"""hyd8f is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyd8f import hyd8f


def test_hyd8f_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hyd8f()
