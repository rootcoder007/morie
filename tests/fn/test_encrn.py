"""encrn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.encrn import encrn


def test_encrn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        encrn()
