"""secle is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.secle import secle


def test_secle_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        secle()
