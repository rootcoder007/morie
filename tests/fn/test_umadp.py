"""umadp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.umadp import umadp


def test_umadp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        umadp()
