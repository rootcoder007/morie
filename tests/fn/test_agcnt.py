"""agcnt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agcnt import agcnt


def test_agcnt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agcnt()
