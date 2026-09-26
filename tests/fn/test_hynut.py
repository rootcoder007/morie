"""hynut is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hynut import hynut


def test_hynut_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hynut()
