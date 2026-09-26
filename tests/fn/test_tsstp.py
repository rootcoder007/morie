"""tsstp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tsstp import tsstp


def test_tsstp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tsstp()
