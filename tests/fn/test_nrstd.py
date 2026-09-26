"""nrstd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nrstd import nrstd


def test_nrstd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nrstd()
