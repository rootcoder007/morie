"""sdmsig is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sdmsig import sdmsig


def test_sdmsig_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sdmsig(resid=None, n=None)
