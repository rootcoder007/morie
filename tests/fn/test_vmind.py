"""vmind is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmind import vmind


def test_vmind_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmind()
