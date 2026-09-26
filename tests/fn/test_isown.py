"""isown is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.isown import isown


def test_isown_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        isown()
