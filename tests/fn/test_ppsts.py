"""ppsts is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppsts import ppsts


def test_ppsts_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppsts()
