"""idpts is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idpts import idpts


def test_idpts_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idpts()
