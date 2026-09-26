"""ptcsr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptcsr import csr_test


def test_ptcsr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csr_test(data=None)
