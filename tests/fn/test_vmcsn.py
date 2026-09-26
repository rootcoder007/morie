"""vmcsn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmcsn import vmcsn


def test_vmcsn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmcsn()
