"""cmopn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cmopn import cmopn


def test_cmopn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cmopn()
