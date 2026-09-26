"""cszon is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cszon import cszon


def test_cszon_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cszon()
