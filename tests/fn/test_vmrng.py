"""vmrng is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmrng import vmrng


def test_vmrng_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmrng()
