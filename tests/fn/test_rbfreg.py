"""rbfreg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rbfreg import rbfreg


def test_rbfreg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rbfreg()
