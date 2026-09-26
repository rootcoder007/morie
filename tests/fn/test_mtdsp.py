"""mtdsp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtdsp import mtdsp


def test_mtdsp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtdsp()
