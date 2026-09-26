"""eneqk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.eneqk import eneqk


def test_eneqk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        eneqk()
