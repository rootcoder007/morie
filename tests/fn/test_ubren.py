"""ubren is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubren import ubren


def test_ubren_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubren()
