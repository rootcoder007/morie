"""ppstr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppstr import ppstr


def test_ppstr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppstr()
