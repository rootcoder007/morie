"""rbfgrd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rbfgrd import rbfgrd


def test_rbfgrd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rbfgrd()
