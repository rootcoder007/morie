"""vmcvl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmcvl import vmcvl


def test_vmcvl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmcvl()
