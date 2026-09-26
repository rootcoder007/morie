"""abend is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.abend import abend


def test_abend_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        abend()
