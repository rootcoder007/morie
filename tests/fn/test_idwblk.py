"""idwblk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idwblk import idwblk


def test_idwblk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idwblk()
