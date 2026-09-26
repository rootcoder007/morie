"""aflch is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.aflch import aflch


def test_aflch_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        aflch()
