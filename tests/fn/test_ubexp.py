"""ubexp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubexp import ubexp


def test_ubexp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubexp()
