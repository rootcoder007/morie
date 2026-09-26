"""nbair is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbair import nbair


def test_nbair_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbair()
