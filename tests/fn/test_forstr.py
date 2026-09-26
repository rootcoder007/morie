"""forstr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.forstr import forstr


def test_forstr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        forstr()
