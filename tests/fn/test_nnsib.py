"""nnsib is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nnsib import nnsib


def test_nnsib_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nnsib()
