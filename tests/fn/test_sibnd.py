"""sibnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sibnd import sibnd


def test_sibnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sibnd()
