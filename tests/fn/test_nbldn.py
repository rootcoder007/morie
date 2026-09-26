"""nbldn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbldn import nbldn


def test_nbldn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbldn()
