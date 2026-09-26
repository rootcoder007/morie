"""aflstk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.aflstk import aflstk


def test_aflstk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        aflstk()
