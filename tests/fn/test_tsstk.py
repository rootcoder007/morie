"""tsstk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tsstk import tsstk


def test_tsstk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tsstk()
