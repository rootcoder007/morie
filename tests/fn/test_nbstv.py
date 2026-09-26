"""nbstv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbstv import nbstv


def test_nbstv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbstv()
