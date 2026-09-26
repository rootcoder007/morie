"""ubthr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubthr import ubthr


def test_ubthr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubthr()
