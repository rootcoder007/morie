"""tbmat is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tbmat import tbmat


def test_tbmat_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tbmat()
