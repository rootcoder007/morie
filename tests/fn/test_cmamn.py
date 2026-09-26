"""cmamn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cmamn import cmamn


def test_cmamn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cmamn()
