"""nblde is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nblde import nblde


def test_nblde_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nblde()
