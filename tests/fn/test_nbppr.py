"""nbppr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbppr import nbppr


def test_nbppr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbppr()
