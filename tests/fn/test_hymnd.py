"""hymnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hymnd import hymnd


def test_hymnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hymnd()
