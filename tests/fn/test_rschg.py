"""rschg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rschg import rschg


def test_rschg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rschg()
