"""turng is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.turng import turng


def test_turng_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        turng()
