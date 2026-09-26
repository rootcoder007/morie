"""vmgau is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmgau import vmgau


def test_vmgau_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmgau()
