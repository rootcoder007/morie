"""uniona is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.uniona import uniona


def test_uniona_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        uniona()
