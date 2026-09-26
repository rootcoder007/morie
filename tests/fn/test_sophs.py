"""sophs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sophs import sophs


def test_sophs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sophs()
