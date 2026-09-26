"""opde is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opde import opde


def test_opde_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opde()
