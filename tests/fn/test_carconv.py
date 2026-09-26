"""carconv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.carconv import carconv


def test_carconv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        carconv(W=None)
