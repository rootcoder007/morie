"""tssrn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tssrn import tssrn


def test_tssrn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tssrn()
