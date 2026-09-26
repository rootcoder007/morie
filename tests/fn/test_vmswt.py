"""vmswt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmswt import vmswt


def test_vmswt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmswt()
