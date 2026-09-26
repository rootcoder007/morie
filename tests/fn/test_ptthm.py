"""ptthm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptthm import thomas_process


def test_ptthm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        thomas_process(data=None)
