"""mstri is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mstri import triangle_ineq


def test_mstri_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        triangle_ineq(data=None)
