"""nbzne is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbzne import nbzne


def test_nbzne_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbzne()
