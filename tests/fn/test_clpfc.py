"""clpfc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clpfc import clpfc


def test_clpfc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clpfc()
