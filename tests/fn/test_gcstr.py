"""gcstr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcstr import gcstr


def test_gcstr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcstr()
