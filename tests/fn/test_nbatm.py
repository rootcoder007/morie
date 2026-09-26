"""nbatm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbatm import nbatm


def test_nbatm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbatm()
