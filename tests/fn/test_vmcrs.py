"""vmcrs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmcrs import vmcrs


def test_vmcrs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmcrs()
