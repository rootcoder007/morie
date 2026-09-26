"""cmstr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cmstr import cmstr


def test_cmstr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cmstr()
