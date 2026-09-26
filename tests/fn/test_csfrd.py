"""csfrd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csfrd import csfrd


def test_csfrd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csfrd()
