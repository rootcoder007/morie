"""seebl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.seebl import seebl


def test_seebl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        seebl()
