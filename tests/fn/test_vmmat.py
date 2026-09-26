"""vmmat is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmmat import vmmat


def test_vmmat_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmmat()
