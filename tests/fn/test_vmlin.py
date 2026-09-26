"""vmlin is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmlin import vmlin


def test_vmlin_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmlin()
