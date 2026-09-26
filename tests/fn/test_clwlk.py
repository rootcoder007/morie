"""clwlk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clwlk import clwlk


def test_clwlk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clwlk()
