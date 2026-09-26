"""adaptg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.adaptg import adaptg


def test_adaptg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        adaptg()
