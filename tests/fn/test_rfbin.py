"""rfbin is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rfbin import rfbin


def test_rfbin_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rfbin()
