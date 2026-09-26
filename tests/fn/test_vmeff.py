"""vmeff is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmeff import vmeff


def test_vmeff_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmeff()
