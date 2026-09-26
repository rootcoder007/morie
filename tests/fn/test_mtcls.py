"""mtcls is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtcls import mtcls


def test_mtcls_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtcls()
