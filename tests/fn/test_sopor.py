"""sopor is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sopor import sopor


def test_sopor_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sopor()
