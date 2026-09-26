"""hsent is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hsent import hsent


def test_hsent_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hsent()
