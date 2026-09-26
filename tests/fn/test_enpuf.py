"""enpuf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.enpuf import enpuf


def test_enpuf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enpuf()
