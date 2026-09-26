"""vmzrn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmzrn import vmzrn


def test_vmzrn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmzrn()
