"""tsstg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tsstg import tsstg


def test_tsstg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tsstg()
