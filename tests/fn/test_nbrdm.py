"""nbrdm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbrdm import nbrdm


def test_nbrdm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbrdm()
