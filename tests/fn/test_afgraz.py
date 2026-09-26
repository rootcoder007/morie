"""afgraz is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afgraz import afgraz


def test_afgraz_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afgraz()
