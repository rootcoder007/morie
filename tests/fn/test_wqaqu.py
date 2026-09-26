"""wqaqu is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqaqu import wqaqu


def test_wqaqu_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqaqu()
