"""gabsn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gabsn import gabsn


def test_gabsn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gabsn()
