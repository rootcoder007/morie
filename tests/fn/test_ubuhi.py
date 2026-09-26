"""ubuhi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubuhi import ubuhi


def test_ubuhi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubuhi()
