"""ppstg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppstg import ppstg


def test_ppstg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppstg()
