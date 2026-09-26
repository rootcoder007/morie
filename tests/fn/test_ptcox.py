"""ptcox is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptcox import cox_process


def test_ptcox_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cox_process(data=None)
