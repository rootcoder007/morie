"""iscat is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.iscat import iscat


def test_iscat_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        iscat()
