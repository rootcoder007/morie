"""vmcvr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmcvr import vmcvr


def test_vmcvr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmcvr()
