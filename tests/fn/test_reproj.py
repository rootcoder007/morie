"""reproj is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.reproj import reproj


def test_reproj_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        reproj()
