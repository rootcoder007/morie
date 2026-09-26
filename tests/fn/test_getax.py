"""getax is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.getax import getax


def test_getax_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        getax()
