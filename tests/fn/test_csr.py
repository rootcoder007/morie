"""csr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csr import csr


def test_csr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csr()
