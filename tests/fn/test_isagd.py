"""isagd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.isagd import isagd


def test_isagd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        isagd()
