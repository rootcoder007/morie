"""dkupd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dkupd import dkupd


def test_dkupd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dkupd()
