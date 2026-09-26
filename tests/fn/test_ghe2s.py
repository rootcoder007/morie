"""ghe2s is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghe2s import ghe2s


def test_ghe2s_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghe2s()
