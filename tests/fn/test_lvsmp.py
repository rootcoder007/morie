"""lvsmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.lvsmp import lvsmp


def test_lvsmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lvsmp()
