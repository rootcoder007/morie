"""vgmdr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgmdr import madogram


def test_vgmdr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        madogram(data=None)
