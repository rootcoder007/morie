"""abter is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.abter import abter


def test_abter_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        abter()
