"""mtacc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtacc import mtacc


def test_mtacc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtacc()
