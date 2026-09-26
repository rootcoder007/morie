"""sfglob is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sfglob import sfglob


def test_sfglob_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sfglob(y=None, X=None, W=None)
