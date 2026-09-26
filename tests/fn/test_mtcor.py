"""mtcor is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtcor import mtcor


def test_mtcor_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtcor()
