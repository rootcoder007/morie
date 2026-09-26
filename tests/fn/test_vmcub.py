"""vmcub is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmcub import vmcub


def test_vmcub_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmcub()
