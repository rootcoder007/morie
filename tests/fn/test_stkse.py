"""stkse is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stkse import stkse


def test_stkse_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stkse()
