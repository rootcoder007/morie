"""nbann is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbann import nbann


def test_nbann_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbann()
