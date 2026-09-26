"""opmxc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opmxc import opmxc


def test_opmxc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opmxc()
