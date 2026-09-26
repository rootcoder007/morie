"""vmprd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmprd import vmprd


def test_vmprd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmprd()
