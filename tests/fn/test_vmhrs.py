"""vmhrs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmhrs import vmhrs


def test_vmhrs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmhrs()
