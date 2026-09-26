"""gpmat is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gpmat import gpmat


def test_gpmat_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gpmat()
