"""nbgrd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbgrd import nbgrd


def test_nbgrd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbgrd()
