"""sawbl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sawbl import sawbl


def test_sawbl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sawbl()
