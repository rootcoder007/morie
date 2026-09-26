"""nbwnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbwnd import nbwnd


def test_nbwnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbwnd()
