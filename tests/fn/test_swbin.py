"""swbin is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.swbin import swbin


def test_swbin_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        swbin(W=None)
