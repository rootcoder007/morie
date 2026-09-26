"""sagjk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sagjk import sagjk


def test_sagjk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sagjk()
