"""fosamp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fosamp import fosamp


def test_fosamp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fosamp()
