"""nbgvb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbgvb import nbgvb


def test_nbgvb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbgvb()
