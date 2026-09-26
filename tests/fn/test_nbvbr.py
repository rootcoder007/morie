"""nbvbr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbvbr import nbvbr


def test_nbvbr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbvbr()
