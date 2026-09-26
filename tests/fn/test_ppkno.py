"""ppkno is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppkno import ppkno


def test_ppkno_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppkno()
