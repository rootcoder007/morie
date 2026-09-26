"""tbbnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tbbnd import tbbnd


def test_tbbnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tbbnd()
