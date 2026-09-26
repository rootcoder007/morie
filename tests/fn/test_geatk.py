"""geatk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.geatk import geatk


def test_geatk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        geatk()
