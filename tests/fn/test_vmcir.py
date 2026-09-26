"""vmcir is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmcir import vmcir


def test_vmcir_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmcir()
