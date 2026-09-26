"""sagjb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sagjb import sagjb


def test_sagjb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sagjb()
