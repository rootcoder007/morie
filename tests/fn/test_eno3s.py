"""eno3s is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.eno3s import eno3s


def test_eno3s_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        eno3s()
