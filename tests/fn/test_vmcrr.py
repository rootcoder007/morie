"""vmcrr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmcrr import vmcrr


def test_vmcrr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmcrr()
