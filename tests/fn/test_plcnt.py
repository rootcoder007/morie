"""plcnt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.plcnt import plcnt


def test_plcnt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        plcnt()
