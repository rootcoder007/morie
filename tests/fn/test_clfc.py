"""clfc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clfc import clfc


def test_clfc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clfc()
