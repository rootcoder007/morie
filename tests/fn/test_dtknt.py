"""dtknt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtknt import dtknt


def test_dtknt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtknt()
