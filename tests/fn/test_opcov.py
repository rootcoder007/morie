"""opcov is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opcov import opcov


def test_opcov_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opcov()
