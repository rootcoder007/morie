"""gcch4 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcch4 import gcch4


def test_gcch4_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcch4()
