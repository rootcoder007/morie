"""nbiso is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbiso import nbiso


def test_nbiso_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbiso()
