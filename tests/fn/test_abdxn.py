"""abdxn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.abdxn import abdxn


def test_abdxn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        abdxn()
