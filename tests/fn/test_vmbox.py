"""vmbox is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmbox import vmbox


def test_vmbox_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmbox()
