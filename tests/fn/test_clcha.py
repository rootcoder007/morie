"""clcha is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clcha import clcha


def test_clcha_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clcha()
