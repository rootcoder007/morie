"""vmfrr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmfrr import vmfrr


def test_vmfrr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmfrr()
