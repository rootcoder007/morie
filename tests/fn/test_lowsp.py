"""lowsp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.lowsp import lowsp


def test_lowsp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lowsp()
