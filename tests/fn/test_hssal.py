"""hssal is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hssal import hssal


def test_hssal_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hssal()
