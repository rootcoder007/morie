"""vmcld is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmcld import vmcld


def test_vmcld_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmcld()
