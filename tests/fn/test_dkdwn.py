"""dkdwn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dkdwn import dkdwn


def test_dkdwn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dkdwn()
