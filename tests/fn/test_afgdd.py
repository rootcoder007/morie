"""afgdd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afgdd import afgdd


def test_afgdd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afgdd()
